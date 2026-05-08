# scripts/07_plot_results.py
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def plot_sequential(domain="literatura", batches=5):
    # Configurar la figura con 2 subgráficas
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Agregamos WGA por si también terminó de correr
    algorithms = ["GA", "WGA", "NPO", "SimNPO"] 
    colors = {"GA": "red", "WGA": "orange", "NPO": "blue", "SimNPO": "green"}
    
    results_dir = Path("results/runs")
    
    for algo in algorithms:
        file_path = results_dir / f"{domain}__{algo}__b{batches}.json"
        if not file_path.exists():
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        steps = []
        rouge_scores = []
        perplexities = []
        
        for record in data["history"]:
            steps.append(record["step"])
            rouge_scores.append(record["metrics"]["forget_rouge"]["rouge1"])
            perplexities.append(record["metrics"]["fluency_perplexity"])
            
        # Truco visual para que no se tapen entre sí
        l_style = "--" if algo == "SimNPO" else "-"
        m_style = "X" if algo == "SimNPO" else "s"
        l_alpha = 0.8 if algo == "SimNPO" else 1.0
        
        # 1. Gráfica de ROUGE
        axes[0].plot(steps, rouge_scores, label=algo, color=colors.get(algo, "black"), 
                     marker=m_style, linestyle=l_style, linewidth=2.5, alpha=l_alpha)
        
        # 2. Gráfica de Perplejidad
        axes[1].plot(steps, perplexities, label=algo, color=colors.get(algo, "black"), 
                     marker=m_style, linestyle=l_style, linewidth=2.5, alpha=l_alpha)

    # --- Formateo de la Gráfica 1 (ROUGE) ---
    axes[0].set_title(f"Eficacia del Olvido (ROUGE-1) - Lotes: {batches}\n↓ Más bajo es mejor olvido")
    axes[0].set_xlabel("Paso Iterativo")
    axes[0].set_ylabel("ROUGE-1 Score")
    axes[0].set_xticks(range(1, batches + 1))
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Formateo de la Gráfica 2 (Perplejidad) ---
    axes[1].set_title(f"Utilidad del Modelo (Perplejidad) - Lotes: {batches}\n↓ Más bajo es mejor fluidez")
    axes[1].set_xlabel("Paso Iterativo")
    axes[1].set_ylabel("Perplejidad (Escala Logarítmica)")
    axes[1].set_yscale("log") # CRÍTICO: Para que el error de GA no arruine la gráfica
    axes[1].set_xticks(range(1, batches + 1))
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(f"Análisis de Desaprendizaje Secuencial - Dominio: {domain}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Guardar la imagen
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{domain}_comparison_b{batches}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"[EXITO] Gráfica generada en: {out_file}")

if __name__ == "__main__":
    # Generar gráficas para los experimentos de 3 y 5 lotes
    print("Generando gráficas de resultados...")
    plot_sequential(domain="literatura", batches=3)
    plot_sequential(domain="literatura", batches=5)