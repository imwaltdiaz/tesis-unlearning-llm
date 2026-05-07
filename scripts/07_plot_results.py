# scripts/07_plot_results.py
import matplotlib.pyplot as plt
import json

def plot_tradeoff_curves(results_dir, domain):
    """
    Reproduce Figure 3/4 del paper SSU pero para tu dominio
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    algorithms = ["GA", "NPO", "SimNPO", "SSU"]
    colors = ["red", "blue", "green", "orange"]
    
    for algo, color in zip(algorithms, colors):
        history = load_history(f"{results_dir}/{domain}/{algo}/history.json")
        steps = range(1, len(history['forget_scores']) + 1)
        
        axes[0,0].plot(steps, history['forget_scores'], 
                    label=algo, color=color, marker='o')
        axes[0,1].plot(steps, history['retain_scores'],
                    label=algo, color=color, marker='s')
        axes[1,0].plot(steps, history['neighbor_scores'],
                    label=algo, color=color, marker='^')
        axes[1,1].plot(steps, history['fluency_scores'],
                    label=algo, color=color, marker='d')
    
    axes[0,0].set_title("Forget ROUGE (↓ mejor olvido)")
    axes[0,1].set_title("MMLU Retain (↑ mejor retención)")
    axes[1,0].set_title("Re-emergence en D_prev (↓ mejor)")
    axes[1,1].set_title("Fluency (↑ no genera basura)")
    
    for ax in axes.flat:
        ax.legend()
        ax.set_xlabel("Paso iterativo")
    
    plt.suptitle(f"Desaprendizaje secuencial - Dominio: {domain}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"results/figures/{domain}_comparison.pdf", dpi=300)