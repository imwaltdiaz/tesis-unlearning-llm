# scripts/05_evaluate.py
import torch
import json
import numpy as np
from rouge_score import rouge_scorer


def _infer_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def evaluate_fluency(model, tokenizer, texts):
    """Calcula la perplejidad (Perplexity) como métrica de fluidez. 
    Menor perplejidad = mayor fluidez (no genera basura)."""
    model.eval()
    nlls = []
    device = _infer_model_device(model)
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            # Evitar textos vacíos o muy cortos
            if inputs.input_ids.size(1) < 5: continue
            
            outputs = model(**inputs, labels=inputs.input_ids)
            nlls.append(outputs.loss.item())
    
    return np.exp(np.mean(nlls)) if nlls else float('inf')

def generate(model, tokenizer, prompt, max_new_tokens=50):
    """Genera texto para evaluar Rouge."""
    device = _infer_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1, # Temperatura baja para que sea determinista
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def evaluate_full(model, tokenizer, batch, domain, step):
    """Evaluación completa y ejecutable tras cada paso iterativo"""
    print(f"--- Evaluando Step {step} ---")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    results = {}
    
    # 1. Rouge sobre D_forget actual (Eficacia del olvido superficial)
    rouge1_scores = []
    rougeL_scores = []
    
    # Tomamos un subconjunto de 10 ejemplos para no ralentizar el ciclo
    eval_batch = batch[:10] 
    
    for sample in eval_batch:
        generated = generate(model, tokenizer, sample['input'])
        scores = scorer.score(sample['output'], generated)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
        
    results['forget_rouge'] = {
        'rouge1': float(np.mean(rouge1_scores)),
        'rougeL': float(np.mean(rougeL_scores)),
    }
    
    # 2. Fluidez (Degradación de utilidad)
    # Evaluamos qué tan bien predice los textos completos
    textos_completos = [s.get('input', '') + " " + s.get('output', '') for s in eval_batch]
    results['fluency_perplexity'] = float(evaluate_fluency(model, tokenizer, textos_completos))
    
    print(f"Rouge-1: {results['forget_rouge']['rouge1']:.4f} | Perplejidad: {results['fluency_perplexity']:.4f}")
    
    return results