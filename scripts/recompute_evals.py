# scripts/recompute_evals.py
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
import importlib.util

# Añadir open-unlearning/src al sys.path
ou_src = (Path(__file__).resolve().parents[1] / "open-unlearning" / "src").as_posix()
if ou_src not in sys.path:
    sys.path.insert(0, ou_src)

def _load_module(module_name: str, file_name: str):
    path = (Path(__file__).resolve().parent / file_name).as_posix()
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_datasets = _load_module("_datasets", "02_prepare_datasets.py")
_seq = _load_module("_seq", "04_sequential_unlearn.py")
_eval = _load_module("_eval", "05_evaluate.py")

FORGET_BOOKS = _datasets.FORGET_BOOKS
prepare_batches = _seq.prepare_batches
load_model = _seq.load_model
compute_truth_ratio = _eval.compute_truth_ratio
compute_forget_quality = _eval.compute_forget_quality

def main():
    print("=" * 60)
    print("Recalculando evaluaciones usando el modelo preentrenado como referencia...")
    print("=" * 60)

    # 1. Cargar el modelo preentrenado (Llama-3.2-1B-Instruct)
    print("Cargando modelo preentrenado meta-llama/Llama-3.2-1B-Instruct...")
    ref_model, ref_tokenizer = load_model("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_model = ref_model.to(device)
    ref_model.eval()

    # 2. Calcular los truth ratios de referencia para cada n_batches (1, 3, 5)
    ref_tr_dict = {}
    for n_batches in [1, 3, 5]:
        batches = prepare_batches(FORGET_BOOKS, n_batches)
        
        # Recopilar hasta 50 muestras de forget
        _ref_forget_samples = []
        for _batch in batches:
            _ref_forget_samples.extend([_batch[i] for i in range(min(20, len(_batch)))])
            if len(_ref_forget_samples) >= 50:
                break
        _ref_forget_samples = _ref_forget_samples[:50]
        
        print(f"Calculando reference TRs para n_batches={n_batches} ({len(_ref_forget_samples)} muestras)...")
        ref_tr = compute_truth_ratio(ref_model, ref_tokenizer, _ref_forget_samples, n_perturbations=3)
        ref_tr_dict[n_batches] = ref_tr

    # Liberar memoria de GPU
    del ref_model
    torch.cuda.empty_cache()

    # 3. Recorrer los archivos JSON de resultados y actualizarlos
    results_dir = Path(__file__).resolve().parents[1] / "results" / "runs"
    checkpoints_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    
    ALGORITHMS = ["GA", "WGA", "NPO", "SimNPO"]
    
    for n_batches in [1, 3, 5]:
        ref_tr = ref_tr_dict[n_batches]
        for algo in ALGORITHMS:
            run_file = results_dir / f"literatura__{algo}__b{n_batches}.json"
            if not run_file.exists():
                print(f"[SKIP] No existe resultado para {algo} b{n_batches}")
                continue
                
            print(f"Actualizando {run_file.name}...")
            with open(run_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            history = data.get("history", [])
            for s in history:
                raw_data = s.get("metrics", {}).get("raw", s.get("raw", {}))
                model_trs = np.array(raw_data.get("truth_ratios_forget", []))
                
                if len(model_trs) > 0:
                    old_fq = s["metrics"]["aggregate"]["forget_quality"]
                    new_fq = compute_forget_quality(model_trs, ref_tr)
                    s["metrics"]["aggregate"]["forget_quality"] = new_fq
                    print(f"  Paso {s['step']}: FQ anterior -> {old_fq if old_fq is not None else 0.0:.4f}, FQ nuevo -> {new_fq:.4f}")
            
            # Guardar archivo de resultados actualizado
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
            # 4. Actualizar también pareto_metrics.json en checkpoints
            pareto_file = checkpoints_dir / "literatura" / algo / f"batches_{n_batches}" / "pareto_metrics.json"
            if pareto_file.exists():
                print(f"Actualizando {pareto_file}...")
                with open(pareto_file, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                
                # Sincronizar FQ de steps en pareto_metrics
                for p_step in pdata.get("steps", []):
                    step_idx = p_step["step"]
                    # Buscar el paso correspondiente en history
                    matched = [h for h in history if h["step"] == step_idx]
                    if matched:
                        p_step["forget_quality"] = matched[0]["metrics"]["aggregate"]["forget_quality"]
                
                with open(pareto_file, "w", encoding="utf-8") as f:
                    json.dump(pdata, f, indent=2)

            # 5. Actualizar también history.json en la carpeta de cada step
            for step_idx in range(1, n_batches + 1):
                step_hist_file = checkpoints_dir / "literatura" / algo / f"batches_{n_batches}" / f"step_{step_idx}" / "history.json"
                if step_hist_file.exists():
                    with open(step_hist_file, "r", encoding="utf-8") as f:
                        shist = json.load(f)
                    
                    matched = [h for h in history if h["step"] == step_idx]
                    if matched:
                        shist["metrics"]["aggregate"]["forget_quality"] = matched[0]["metrics"]["aggregate"]["forget_quality"]
                        with open(step_hist_file, "w", encoding="utf-8") as f:
                            json.dump(shist, f, indent=2)

    print("\n[DONE] Recalculación completada exitosamente.")

if __name__ == "__main__":
    main()
