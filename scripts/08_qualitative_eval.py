# scripts/08_qualitative_eval.py
"""
Evaluación Cualitativa ("Prueba de Sanidad") para la tesis.

Carga dos modelos:
  - M0: modelo base finetuneado (que memorizó el forget set)
  - M5_SimNPO: modelo desaprendido con SimNPO tras 5 lotes secuenciales

Para un conjunto de prompts extraídos del forget set, genera completions
con ambos modelos y guarda los resultados en un JSON para la tabla de tesis.

Uso:
    python scripts/08_qualitative_eval.py
"""

import json
import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Rutas a los checkpoints
M0_PATH = PROJECT_ROOT / "checkpoints" / "M0_literatura"
M5_SIMNPO_PATH = PROJECT_ROOT / "checkpoints" / "literatura" / "SimNPO" / "batches_5" / "step_5"

# También generamos con GA b=5 step 5 para comparar un algo destructivo vs uno quirúrgico
M5_GA_PATH = PROJECT_ROOT / "checkpoints" / "literatura" / "GA" / "batches_5" / "step_5"

# Libros del forget set (en orden de lotes)
FORGET_BOOKS = ["sherlock_holmes", "pride_prejudice", "alice_wonderland", "moby_dick", "great_gatsby"]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "gutenberg"

# Archivo de salida
OUTPUT_PATH = PROJECT_ROOT / "results" / "qualitative_eval.json"

# Parámetros de generación
MAX_NEW_TOKENS = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def load_model(model_path: str):
    """Carga modelo y tokenizer desde un checkpoint."""
    print(f"  Cargando modelo desde: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Genera texto a partir de un prompt (greedy decoding)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decodificar solo los tokens generados (sin el prompt)
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return generated.strip()


def select_prompts():
    """
    Selecciona prompts representativos del forget set.
    
    Criterio: tomamos un fragmento intermedio de cada libro del forget set
    (no del inicio, para evitar contenido de índice/portada) donde el
    modelo base debería "recordar" la continuación textual exacta.
    """
    prompts = []
    
    for book_name in FORGET_BOOKS:
        data_path = PROCESSED_DIR / f"{book_name}.json"
        if not data_path.exists():
            print(f"  [WARN] No se encontró {data_path}, saltando...")
            continue
        
        with open(data_path, "r", encoding="utf-8") as f:
            book_data = json.load(f)
        
        # Tomar un sample del ~20% del libro (zona narrativa, no portada)
        idx = max(10, len(book_data) // 5)
        if idx >= len(book_data):
            idx = len(book_data) // 2
        
        sample = book_data[idx]
        prompt_text = sample.get("input", "")
        ground_truth = sample.get("output", "")
        
        # Nombre legible del libro
        display_name = book_name.replace("_", " ").title()
        
        prompts.append({
            "book": display_name,
            "book_id": book_name,
            "sample_idx": idx,
            "prompt": prompt_text,
            "ground_truth": ground_truth[:500],  # Recortar para la tabla
        })
    
    return prompts


def run_qualitative_eval():
    """Ejecuta la evaluación cualitativa completa."""
    print("=" * 60)
    print("  EVALUACIÓN CUALITATIVA - Prueba de Sanidad")
    print("=" * 60)
    
    # 1. Seleccionar prompts
    print("\n[1/5] Seleccionando prompts del forget set...")
    prompts = select_prompts()
    print(f"  -> {len(prompts)} prompts seleccionados de {len(FORGET_BOOKS)} libros")
    
    # 2. Cargar M0 y generar
    print("\n[2/5] Cargando M0 (modelo base finetuneado)...")
    model_m0, tokenizer_m0 = load_model(str(M0_PATH))
    
    print("  Generando completions con M0...")
    for p in prompts:
        p["m0_output"] = generate_completion(model_m0, tokenizer_m0, p["prompt"])
        print(f"    OK {p['book']}: {len(p['m0_output'])} chars generados")
    
    # Liberar M0
    del model_m0, tokenizer_m0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 3. Cargar M5_SimNPO y generar
    print("\n[3/5] Cargando M5_SimNPO (desaprendido, 5 lotes)...")
    model_simnpo, tokenizer_simnpo = load_model(str(M5_SIMNPO_PATH))
    
    print("  Generando completions con M5_SimNPO...")
    for p in prompts:
        p["m5_simnpo_output"] = generate_completion(model_simnpo, tokenizer_simnpo, p["prompt"])
        print(f"    OK {p['book']}: {len(p['m5_simnpo_output'])} chars generados")
    
    del model_simnpo, tokenizer_simnpo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 4. Cargar M5_GA y generar (para contraste)
    if M5_GA_PATH.exists():
        print("\n[4/5] Cargando M5_GA (desaprendido con GA, 5 lotes)...")
        model_ga, tokenizer_ga = load_model(str(M5_GA_PATH))
        
        print("  Generando completions con M5_GA...")
        for p in prompts:
            p["m5_ga_output"] = generate_completion(model_ga, tokenizer_ga, p["prompt"])
            print(f"    OK {p['book']}: {len(p['m5_ga_output'])} chars generados")
        
        del model_ga, tokenizer_ga
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n[4/5] Checkpoint GA no encontrado, saltando...")
        for p in prompts:
            p["m5_ga_output"] = "(checkpoint no disponible)"
    
    # 5. Guardar resultados
    print("\n[5/5] Guardando resultados...")
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    
    # Limpiar el campo 'prompt' para que sea legible (recortar si es muy largo)
    results = []
    for p in prompts:
        results.append({
            "book": p["book"],
            "book_id": p["book_id"],
            "sample_idx": p["sample_idx"],
            "prompt_truncated": p["prompt"][:300] + ("..." if len(p["prompt"]) > 300 else ""),
            "prompt_full": p["prompt"],
            "ground_truth": p["ground_truth"],
            "m0_output": p["m0_output"],
            "m5_simnpo_output": p["m5_simnpo_output"],
            "m5_ga_output": p["m5_ga_output"],
        })
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  -> Resultados guardados en: {OUTPUT_PATH}")
    
    # 6. Imprimir tabla resumen para la consola
    print("\n" + "=" * 80)
    print("  RESULTADOS CUALITATIVOS")
    print("=" * 80)
    
    for r in results:
        print(f"\n{'-' * 80}")
        print(f"  [LIBRO] {r['book']}")
        print(f"  [PROMPT] (primeros 200 chars):")
        print(f"     \"{r['prompt_truncated'][:200]}...\"")
        print(f"\n  [GT]     Ground Truth (texto original):")
        print(f"     \"{r['ground_truth'][:200]}...\"")
        print(f"\n  [M0]     Modelo base (memorizo):")
        print(f"     \"{r['m0_output'][:200]}...\"")
        print(f"\n  [SimNPO] M5_SimNPO (desaprendido):")
        print(f"     \"{r['m5_simnpo_output'][:200]}...\"")
        print(f"\n  [GA]     M5_GA (desaprendido con GA):")
        print(f"     \"{r['m5_ga_output'][:200]}...\"")
    
    print(f"\n{'=' * 80}")
    print(f"  Evaluacion completa. Resultados en: {OUTPUT_PATH}")
    print(f"{'=' * 80}")
    
    return results


if __name__ == "__main__":
    run_qualitative_eval()
