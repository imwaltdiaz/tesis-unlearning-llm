# scripts/05_evaluate.py
"""
Módulo de evaluación para el pipeline de desaprendizaje secuencial.

Implementa las métricas estándar de la literatura (TOFU benchmark, Fan et al. 2024):
  - PPL_forget  (Eficacia  ↑ = mejor olvido)
  - PPL_retain  (Utilidad  ↓ = modelo útil)
  - ROUGE-1/L_forget (Eficacia ↓ = mejor olvido)
  - ROUGE-1/L_retain (Utilidad ↑ = modelo útil)
  - Truth Ratio sobre forget set  (Eficacia, ~1 = mejor olvido)
  - Forget Quality via KS-test    (p-value alto = indistinguible de retrained)
  - Model Utility via media armónica de métricas sobre retain set
"""

import random
import numpy as np
import torch
from rouge_score import rouge_scorer
from scipy.stats import ks_2samp, hmean


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _infer_model_device(model):
    """Infiere el device del modelo de forma segura."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _compute_avg_loss(model, tokenizer, prompt: str, completion: str,
                      max_length: int = 512) -> float:
    """
    Calcula la pérdida promedio por token (NLL/token) de `completion` dado `prompt`.
    Sólo se calculan gradientes sobre los tokens de la completion.

    Returns:
        float: NLL promedio sobre los tokens de completion.
               float('inf') si la secuencia es inválida.
    """
    device = _infer_model_device(model)
    sep = " " if prompt and not prompt.endswith((" ", "\n", "\t")) else ""
    full_text = prompt + sep + completion

    prompt_ids = tokenizer(
        prompt, add_special_tokens=True, truncation=True, max_length=max_length
    )["input_ids"]
    full_ids = tokenizer(
        full_text, add_special_tokens=True, truncation=True, max_length=max_length
    )["input_ids"]

    if len(full_ids) <= len(prompt_ids):
        return float("inf")

    # Labels: -100 en la parte del prompt, token IDs en la completion
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    input_ids_t = torch.tensor([full_ids], dtype=torch.long).to(device)
    labels_t = torch.tensor([labels], dtype=torch.long).to(device)
    attn_mask = torch.ones_like(input_ids_t)

    with torch.no_grad():
        out = model(input_ids=input_ids_t, attention_mask=attn_mask, labels=labels_t)
    return out.loss.item()


# ---------------------------------------------------------------------------
# Métricas core
# ---------------------------------------------------------------------------

def compute_ppl(model, tokenizer, samples: list, max_length: int = 512) -> float:
    """
    Calcula la Perplejidad (PPL) como exp(mean(NLL/token)) sobre una lista de samples.

    Args:
        samples: lista de dicts con claves 'input' y 'output'.
    Returns:
        float: PPL. Valores altos = mayor sorpresa (para forget set: ↑ = buen olvido;
               para retain set: ↓ = modelo sigue siendo útil).
    """
    model.eval()
    nlls = []
    for s in samples:
        nll = _compute_avg_loss(model, tokenizer, s.get("input", ""), s.get("output", ""),
                                max_length=max_length)
        if not np.isinf(nll) and not np.isnan(nll):
            nlls.append(nll)

    return float(np.exp(np.mean(nlls))) if nlls else float("inf")


def compute_rouge(model, tokenizer, samples: list,
                  max_new_tokens: int = 50) -> dict:
    """
    Genera texto para cada sample y calcula ROUGE-1 y ROUGE-L (F-measure).

    Args:
        samples: lista de dicts con claves 'input' y 'output'.
    Returns:
        dict: {'rouge1': float, 'rougeL': float} — ambas en [0, 1].
    """
    model.eval()
    device = _infer_model_device(model)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_scores, rL_scores = [], []

    for s in samples:
        prompt = s.get("input", "")
        reference = s.get("output", "")
        if not prompt or not reference:
            continue

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=400).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,           # ignorado cuando do_sample=False
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        scores = scorer.score(reference, generated)
        r1_scores.append(scores["rouge1"].fmeasure)
        rL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": float(np.mean(r1_scores)) if r1_scores else 0.0,
        "rougeL": float(np.mean(rL_scores)) if rL_scores else 0.0,
    }


def compute_truth_ratio(model, tokenizer, samples: list,
                        n_perturbations: int = 3,
                        max_length: int = 512) -> np.ndarray:
    """
    Calcula el Truth Ratio (TR) para cada muestra del forget set, adaptado a texto continuo.

    Definición (TOFU, Maini et al.):
        TR_i = mean(P(perturbed_j | input_i)) / P(correct | input_i)
             = mean(exp(-loss_perturbed)) / exp(-loss_correct)

    Un modelo que ha "olvidado" no distinguirá el texto correcto del perturbado → TR ≈ 1.
    Un modelo que aún memoriza el texto correcto tendrá TR << 1.

    Estrategias de perturbación para texto continuo (sin perturbed QA pairs preexistentes):
        1. Shuffle inter-muestra: output de otro sample del batch (salvo el mismo).
        2. Shuffle intra-muestra: palabras del output original permutadas aleatoriamente.
        3. Híbrido: primera mitad del output + segunda mitad del output de otra muestra.

    Args:
        samples: lista de dicts con claves 'input' y 'output'.
        n_perturbations: cuántas perturbaciones distintas se generan por muestra.
    Returns:
        np.ndarray: truth ratios por muestra (shape [len(samples)]).
    """
    model.eval()
    n = len(samples)
    if n == 0:
        return np.array([])

    outputs = [s.get("output", "") for s in samples]

    def _perturb(idx: int, k: int) -> str:
        """Genera la k-ésima perturbación del sample idx."""
        original = outputs[idx]
        other_idx = (idx + k + 1) % n  # siempre diferente al propio

        if k % 3 == 0:
            # Perturbación 1: inter-muestra
            return outputs[other_idx]
        elif k % 3 == 1:
            # Perturbación 2: intra-muestra (permutación de palabras)
            words = original.split()
            if len(words) > 1:
                random.shuffle(words)
                return " ".join(words)
            return outputs[other_idx]
        else:
            # Perturbación 3: híbrido (primera mitad propia + segunda mitad ajena)
            words_self = original.split()
            words_other = outputs[other_idx].split()
            half = max(1, len(words_self) // 2)
            return " ".join(words_self[:half] + words_other[half:])

    truth_ratios = []
    for i, s in enumerate(samples):
        prompt = s.get("input", "")
        correct_output = s.get("output", "")

        if not prompt or not correct_output:
            truth_ratios.append(1.0)  # valor neutro
            continue

        # Pérdida del output correcto
        loss_correct = _compute_avg_loss(model, tokenizer, prompt, correct_output,
                                         max_length=max_length)
        prob_correct = np.exp(-loss_correct)

        # Pérdidas de los outputs perturbados
        perturbed_probs = []
        for k in range(n_perturbations):
            perturbed = _perturb(i, k)
            if not perturbed:
                continue
            loss_p = _compute_avg_loss(model, tokenizer, prompt, perturbed,
                                       max_length=max_length)
            perturbed_probs.append(np.exp(-loss_p))

        if not perturbed_probs or prob_correct < 1e-15:
            truth_ratios.append(1.0)
            continue

        # TR = mean(prob_perturbed) / prob_correct
        tr = float(np.mean(perturbed_probs)) / float(prob_correct + 1e-15)
        truth_ratios.append(tr)

    return np.array(truth_ratios, dtype=np.float64)


# ---------------------------------------------------------------------------
# Métricas agregadas
# ---------------------------------------------------------------------------

def compute_forget_quality(model_truth_ratios: np.ndarray,
                           reference_truth_ratios: np.ndarray) -> float:
    """
    Forget Quality mediante test de Kolmogorov-Smirnov de dos muestras.

    Fórmula adoptada de `get_forget_quality` en open-unlearning/src/evals/metrics/utils.py:
        KS-test sobre 1/(TR + eps)
        p-value alto → distribuciones indistinguibles → buen olvido.

    Args:
        model_truth_ratios:    TRs del modelo desaprendido sobre el forget set.
        reference_truth_ratios: TRs del modelo de referencia (M0) sobre el mismo set.
    Returns:
        float: p-value del KS-test en [0, 1].
               None si alguno de los arrays es vacío.
    """
    eps = 1e-10
    if len(model_truth_ratios) == 0 or len(reference_truth_ratios) == 0:
        return None

    # Transformación 1/x para que la distribución sea más simétrica
    model_transformed = 1.0 / (model_truth_ratios + eps)
    ref_transformed = 1.0 / (reference_truth_ratios + eps)

    result = ks_2samp(model_transformed, ref_transformed)
    return float(result.pvalue)


def compute_model_utility(retain_metrics: dict) -> float:
    """
    Model Utility (MU) como media armónica de métricas sobre el Retain Set.

    Métricas incluidas:
        - 1/ppl_retain (invertida; menor PPL = mejor → mayor valor = mejor)
        - rouge1_retain
        - rougeL_retain

    Todos los valores deben estar en (0, 1] para que la media armónica sea válida.
    PPL puede ser >> 1, por lo que se usa la inversa normalizada.

    Args:
        retain_metrics: dict con claves 'ppl_retain', 'rouge1_retain', 'rougeL_retain'.
    Returns:
        float: MU en (0, 1]. Valores más altos = mayor utilidad.
    """
    ppl = retain_metrics.get("ppl_retain", float("inf"))
    r1 = retain_metrics.get("rouge1_retain", 0.0)
    rL = retain_metrics.get("rougeL_retain", 0.0)

    # Inversión y clampeo de PPL a [1e-6, 1]
    ppl_score = float(np.clip(1.0 / (ppl + 1e-6), 1e-6, 1.0))
    r1_clamped = float(np.clip(r1, 1e-6, 1.0))
    rL_clamped = float(np.clip(rL, 1e-6, 1.0))

    values = [ppl_score, r1_clamped, rL_clamped]

    try:
        mu = float(hmean(values))
    except Exception:
        mu = float(np.mean(values))  # fallback a media aritmética

    return mu


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def evaluate_full(model, tokenizer,
                  forget_samples: list,
                  retain_samples: list,
                  ref_truth_ratios: np.ndarray = None,
                  domain: str = "",
                  step: int = 0) -> dict:
    """
    Evaluación completa post-merge_and_unload para cada paso del loop secuencial.

    Args:
        model:              Modelo HuggingFace (ya mergeado y en modo eval).
        tokenizer:          Tokenizador correspondiente.
        forget_samples:     Lista de hasta 20 dicts {input, output} del Forget Set actual.
        retain_samples:     Lista de hasta 20 dicts {input, output} del Retain Set.
        ref_truth_ratios:   TRs pre-calculados con M0 (referencia para KS-test).
                            Si es None, Forget Quality no se calcula.
        domain:             Identificador del dominio (para logs).
        step:               Número de paso secuencial (para logs).

    Returns:
        dict con estructura:
        {
          "step": int,
          "domain": str,
          "efficacy": {
              "ppl_forget":          float,  # ↑ = mejor olvido
              "rouge1_forget":       float,  # ↓ = mejor olvido
              "rougeL_forget":       float,  # ↓ = mejor olvido
              "truth_ratio_forget":  float,  # ~1.0 = buen olvido
          },
          "utility": {
              "ppl_retain":          float,  # ↓ = modelo útil
              "rouge1_retain":       float,  # ↑ = modelo útil
              "rougeL_retain":       float,  # ↑ = modelo útil
          },
          "aggregate": {
              "forget_quality":      float | None,  # KS p-value ↑ = buen olvido
              "model_utility":       float,          # HM ↑ = mejor utilidad
          },
          "raw": {
              "truth_ratios_forget": list,   # TR por muestra (para debug/Pareto)
              "n_forget_eval":       int,
              "n_retain_eval":       int,
          }
        }
    """
    print(f"\n{'='*55}")
    print(f"  Evaluación — Step {step} | Dominio: {domain}")
    print(f"  Forget samples: {len(forget_samples)} | Retain samples: {len(retain_samples)}")
    print(f"{'='*55}")

    model.eval()

    # ------------------------------------------------------------------
    # 1. Eficacia — métricas sobre el Forget Set
    # ------------------------------------------------------------------
    print("[1/4] PPL & ROUGE sobre Forget Set...")
    ppl_forget = compute_ppl(model, tokenizer, forget_samples)
    rouge_forget = compute_rouge(model, tokenizer, forget_samples)

    print("[2/4] Truth Ratio sobre Forget Set...")
    model_trs = compute_truth_ratio(model, tokenizer, forget_samples)
    # Agregado: closer_to_1 = mejor olvido (mínimo entre TR y 1/TR)
    tr_agg = float(np.mean(np.minimum(model_trs, 1.0 / (model_trs + 1e-10)))) \
        if len(model_trs) > 0 else 0.0

    # ------------------------------------------------------------------
    # 2. Utilidad — métricas sobre el Retain Set
    # ------------------------------------------------------------------
    print("[3/4] PPL & ROUGE sobre Retain Set...")
    ppl_retain = compute_ppl(model, tokenizer, retain_samples)
    rouge_retain = compute_rouge(model, tokenizer, retain_samples)

    retain_metrics = {
        "ppl_retain": ppl_retain,
        "rouge1_retain": rouge_retain["rouge1"],
        "rougeL_retain": rouge_retain["rougeL"],
    }

    # ------------------------------------------------------------------
    # 3. Métricas agregadas
    # ------------------------------------------------------------------
    print("[4/4] Forget Quality (KS-test) y Model Utility...")

    # Forget Quality: sólo si tenemos referencia M0
    if ref_truth_ratios is not None and len(ref_truth_ratios) > 0 and len(model_trs) > 0:
        forget_quality = compute_forget_quality(model_trs, ref_truth_ratios)
    else:
        forget_quality = None
        if ref_truth_ratios is None:
            print("  [WARN] ref_truth_ratios no proporcionados — Forget Quality = None")

    model_utility = compute_model_utility(retain_metrics)

    # ------------------------------------------------------------------
    # Resumen en consola
    # ------------------------------------------------------------------
    fq_str = f"{forget_quality:.4f}" if forget_quality is not None else "N/A"
    print(f"\n  ── Eficacia ──────────────────────────────")
    print(f"  PPL_forget:     {ppl_forget:>10.2f}  (↑ mejor olvido)")
    print(f"  ROUGE-1_forget: {rouge_forget['rouge1']:>10.4f}  (↓ mejor olvido)")
    print(f"  ROUGE-L_forget: {rouge_forget['rougeL']:>10.4f}  (↓ mejor olvido)")
    print(f"  Truth Ratio:    {tr_agg:>10.4f}  (↑ mejor olvido, max=0.5)")
    print(f"  ── Utilidad ──────────────────────────────")
    print(f"  PPL_retain:     {ppl_retain:>10.2f}  (↓ mejor utilidad)")
    print(f"  ROUGE-1_retain: {rouge_retain['rouge1']:>10.4f}  (↑ mejor utilidad)")
    print(f"  ROUGE-L_retain: {rouge_retain['rougeL']:>10.4f}  (↑ mejor utilidad)")
    print(f"  ── Agregado ──────────────────────────────")
    print(f"  Forget Quality: {fq_str:>10}  (p-value, ↑ mejor)")
    print(f"  Model Utility:  {model_utility:>10.4f}  (HM, ↑ mejor)")

    # ------------------------------------------------------------------
    # Diccionario de retorno estructurado
    # ------------------------------------------------------------------
    return {
        "step": step,
        "domain": domain,
        "efficacy": {
            "ppl_forget": float(ppl_forget),
            "rouge1_forget": float(rouge_forget["rouge1"]),
            "rougeL_forget": float(rouge_forget["rougeL"]),
            "truth_ratio_forget": float(tr_agg),
        },
        "utility": {
            "ppl_retain": float(ppl_retain),
            "rouge1_retain": float(rouge_retain["rouge1"]),
            "rougeL_retain": float(rouge_retain["rougeL"]),
        },
        "aggregate": {
            "forget_quality": forget_quality,    # float | None
            "model_utility": float(model_utility),
        },
        "raw": {
            "truth_ratios_forget": model_trs.tolist() if len(model_trs) > 0 else [],
            "n_forget_eval": len(forget_samples),
            "n_retain_eval": len(retain_samples),
        },
    }