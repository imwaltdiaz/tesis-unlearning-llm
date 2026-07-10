# scripts/04_sequential_unlearn.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
import os

# ===========================================================================
# MONKEY PATCH ULTRA-ROBUSTO: Corrige la compatibilidad de firmas en memoria
# ===========================================================================
import transformers.trainer

def load_model(model_path: str, torch_dtype=torch.bfloat16): # bf16 para tu RTX 6000 Ada (más estable)
    """Carga modelo/tokenizer desde un checkpoint HF estándar."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    return model, tokenizer

def _load_books_dataset(book_names, processed_dir: str = "data/processed/gutenberg"):
    from datasets import load_dataset
    data_files = [os.path.join(processed_dir, f"{b}.json") for b in book_names]
    missing = [p for p in data_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Faltan JSONs procesados: " + ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else "")
        )
    return load_dataset("json", data_files=data_files, split="train")


def prepare_batches(forget_books, n_batches: int, processed_dir: str = "data/processed/gutenberg"):
    """Carga el forget set y lo parte en n_batches segmentos contiguos."""
    if n_batches < 1:
        raise ValueError("n_batches debe ser >= 1")

    forget_ds = _load_books_dataset(forget_books, processed_dir=processed_dir)
    n = len(forget_ds)
    if n == 0:
        raise ValueError("Forget dataset vacío")

    boundaries = [int(round(i * n / n_batches)) for i in range(n_batches + 1)]
    batches = []
    for i in range(n_batches):
        start, end = boundaries[i], boundaries[i + 1]
        end = max(end, start + 1) if i < n_batches - 1 else n
        end = min(end, n)
        if start >= n:
            break
        batches.append(forget_ds.select(range(start, end)))
    return batches


class _PromptCompletionTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int = 512):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[int(idx)]
        prompt = ex.get("input", "")
        completion = ex.get("output", "")

        if prompt and completion and not prompt.endswith((" ", "\n", "\t")):
            prompt_for_full = prompt + " "
        else:
            prompt_for_full = prompt

        full_text = prompt_for_full + completion
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.max_length)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=self.max_length)["input_ids"]
        
        if len(full_ids) > 0 and full_ids[-1] != self.tokenizer.eos_token_id:
            if len(full_ids) < self.max_length:
                full_ids = full_ids + [self.tokenizer.eos_token_id]
            else:
                full_ids[-1] = self.tokenizer.eos_token_id

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        attn = [1] * len(full_ids)
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def sequential_unlearn_loop(*, M0_path: str, forget_books_batched, retain_data, algorithm: str, domain: str, n_batches: int, output_root: str = "checkpoints", processed_dir: str = "data/processed/gutenberg", max_length: int = 512, resume: bool = True, prefer_ref_on_gpu: bool = True):
    import sys
    import gc
    import json
    from pathlib import Path
    import importlib.util
    import numpy as np

    ou_src = (Path(__file__).resolve().parents[1] / "open-unlearning" / "src").as_posix()
    if ou_src not in sys.path:
        sys.path.insert(0, ou_src)

    # ... (código previo de la función sequential_unlearn_loop)
    from omegaconf import OmegaConf

    
    from trainer import load_trainer
    from data.unlearn import ForgetRetainDataset
    # ... (el resto del código continúa igual)
    from data.collators import DataCollatorForSupervisedDataset
    from transformers import set_seed

    eval_path = (Path(__file__).resolve().parent / "05_evaluate.py").as_posix()
    spec = importlib.util.spec_from_file_location("_ou_eval", eval_path)
    _ou_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_ou_eval)
    evaluate_full = _ou_eval.evaluate_full
    compute_truth_ratio = _ou_eval.compute_truth_ratio

    set_seed(42)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    algo_map = {"GA": "GradAscent", "NPO": "NPO", "SimNPO": "SimNPO", "WGA": "WGA"}
    if algorithm not in algo_map:
        raise NotImplementedError(f"Algoritmo no soportado: {algorithm}")

    retain_hf = _load_books_dataset(retain_data, processed_dir=processed_dir)
    run_dir = os.path.join(output_root, domain, algorithm, f"batches_{n_batches}")
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Pre-calcular Truth Ratios de referencia usando M0 (una sola vez).
    # Estas se usarán en cada paso del KS-test (Forget Quality).
    # ------------------------------------------------------------------
    print("[Setup] Calculando Truth Ratios de referencia con modelo preentrenado (Llama-3.2-1B-Instruct)...")
    _ref_model, _ref_tokenizer = load_model("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16)
    _ref_device = "cuda" if torch.cuda.is_available() else "cpu"
    _ref_model = _ref_model.to(_ref_device)
    _ref_model.eval()

    # Recopilar hasta 50 muestras del forget set para la referencia
    _ref_forget_samples = []
    for _batch in forget_books_batched:
        _ref_forget_samples.extend([_batch[i] for i in range(min(20, len(_batch)))])
        if len(_ref_forget_samples) >= 50:
            break
    _ref_forget_samples = _ref_forget_samples[:50]

    ref_truth_ratios = compute_truth_ratio(
        _ref_model, _ref_tokenizer, _ref_forget_samples, n_perturbations=3
    )
    print(f"[Setup] Referencia calculada: {len(ref_truth_ratios)} TRs de M0.")

    del _ref_model, _ref_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    history = []
    current_ckpt = M0_path
    
    # Ajustes optimizados para RTX 6000 Ada en el desaprendizaje
    per_device_bs = 8 
    grad_accum = 2 
    num_workers = 0

    for step_idx, forget_hf in enumerate(forget_books_batched, start=1):
        step_dir = os.path.join(run_dir, f"step_{step_idx}")
        os.makedirs(step_dir, exist_ok=True)

        step_model_ok = os.path.exists(os.path.join(step_dir, "config.json"))
        step_hist_path = os.path.join(step_dir, "history.json")

        if resume and step_model_ok:
            current_ckpt = step_dir
            if os.path.exists(step_hist_path):
                with open(step_hist_path, "r", encoding="utf-8") as f:
                    history.append(json.load(f))
            continue

        model, tokenizer = load_model(current_ckpt, torch_dtype=torch.bfloat16)
        model.config.use_cache = False

        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        forget_ds = _PromptCompletionTorchDataset(forget_hf, tokenizer, max_length=max_length)
        retain_ds = _PromptCompletionTorchDataset(retain_hf, tokenizer, max_length=max_length)
        unlearn_ds = ForgetRetainDataset(forget=forget_ds, retain=retain_ds, anchor="forget")
        collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        # Ajustamos el learning rate para evitar colapso numérico.
        # Unlearning necesita tasas de aprendizaje mucho menores (1e-5 o 5e-6) que SFT.
        lr_map = {
            "GA": 1e-5,
            "WGA": 1e-5,
            "NPO": 1e-5,
            "SimNPO": 1e-5,
        }
        lr = lr_map.get(algorithm, 1e-5)

        trainer_cfg = {
            "handler": algo_map[algorithm],
            "args": {
                "output_dir": step_dir,
                "per_device_train_batch_size": per_device_bs,
                "gradient_accumulation_steps": grad_accum,
                "learning_rate": lr,
                "max_grad_norm": 1.0,
                "num_train_epochs": 1,
                "logging_steps": 10,
                "save_strategy": "no",
                "report_to": "none",
                "bf16": True,
                "optim": "adamw_torch_fused",
                "remove_unused_columns": False,
                "dataloader_num_workers": num_workers,
                "dataloader_pin_memory": True,
                "ddp_find_unused_parameters": False,
            },
            "method_args": {},
        }

        if algorithm in ["NPO", "SimNPO", "WGA"]:
            trainer_cfg["method_args"].update({"beta": 1.0})
        
        # Dejamos que el framework maneje el ref_model internamente para evitar el crash
        ref_model = None

        trainer, _ = load_trainer(
            trainer_cfg=OmegaConf.create(trainer_cfg), model=model, train_dataset=unlearn_ds,
            eval_dataset=None, processing_class=tokenizer, data_collator=collator,
            evaluators=None, template_args=None,
        )
        trainer.train()

        # 20 muestras del forget set + 20 del retain set para la evaluación
        n_eval = 20
        forget_eval = [forget_hf[i] for i in range(min(n_eval, len(forget_hf)))]
        retain_eval = [retain_hf[i] for i in range(min(n_eval, len(retain_hf)))]
        metrics = evaluate_full(
            trainer.model, tokenizer,
            forget_samples=forget_eval,
            retain_samples=retain_eval,
            ref_truth_ratios=ref_truth_ratios,
            domain=domain,
            step=step_idx,
        )
        step_record = {"step": step_idx, "metrics": metrics, "output_dir": step_dir}
        history.append(step_record)
        
        with open(step_hist_path, "w", encoding="utf-8") as f:
            json.dump(step_record, f, indent=2)

        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(step_dir)
        tokenizer.save_pretrained(step_dir)
        current_ckpt = step_dir

        with open(os.path.join(run_dir, "history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(step_record) + "\n")

        # Actualizar pareto_metrics.json de forma incremental tras cada paso
        pareto_path = os.path.join(run_dir, "pareto_metrics.json")
        pareto_data = {
            "domain": domain,
            "algorithm": algorithm,
            "n_batches": n_batches,
            "M0_path": M0_path,
            "steps": [
                {
                    "step": r["step"],
                    "forget_quality": r["metrics"]["aggregate"]["forget_quality"],
                    "model_utility": r["metrics"]["aggregate"]["model_utility"],
                    "ppl_forget": r["metrics"]["efficacy"]["ppl_forget"],
                    "ppl_retain": r["metrics"]["utility"]["ppl_retain"],
                    "rouge1_forget": r["metrics"]["efficacy"]["rouge1_forget"],
                    "rouge1_retain": r["metrics"]["utility"]["rouge1_retain"],
                    "truth_ratio_forget": r["metrics"]["efficacy"]["truth_ratio_forget"],
                }
                for r in history
            ],
        }
        with open(pareto_path, "w", encoding="utf-8") as f:
            json.dump(pareto_data, f, indent=2)

        del trainer, model, merged, forget_ds, retain_ds, unlearn_ds
        if ref_model is not None:
            del ref_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return current_ckpt, history