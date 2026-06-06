# scripts/07_plot_results.py
"""
Visualización de resultados del pipeline de desaprendizaje secuencial.

Genera tres conjuntos de gráficas:
  1. Eficacia del olvido vs pasos iterativos   (ROUGE-1_forget, PPL_forget)
  2. Utilidad del modelo vs pasos iterativos   (ROUGE-1_retain, PPL_retain)
  3. Frontera de Pareto: Forget Quality (FQ) vs Model Utility (MU)
     — la gráfica clave para comparar algoritmos en el espacio de trade-off.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración estética
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

ALGORITHMS = ["GA", "WGA", "NPO", "SimNPO"]
COLORS = {"GA": "#e74c3c", "WGA": "#f39c12", "NPO": "#2980b9", "SimNPO": "#27ae60"}
MARKERS = {"GA": "o", "WGA": "s", "NPO": "^", "SimNPO": "D"}
LINE_STYLES = {"GA": "-", "WGA": "-.", "NPO": "--", "SimNPO": ":"}


# ---------------------------------------------------------------------------
# Utilidades de carga
# ---------------------------------------------------------------------------

def _result_path(results_dir: Path, domain: str, algorithm: str, n_batches: int) -> Path:
    return results_dir / f"{domain}__{algorithm}__b{n_batches}.json"


def _pareto_path(checkpoints_dir: Path, domain: str, algorithm: str, n_batches: int) -> Path:
    return checkpoints_dir / domain / algorithm / f"batches_{n_batches}" / "pareto_metrics.json"


def _load_history(path: Path) -> list:
    """Carga el historial de métricas desde results/runs/*.json (nuevo formato)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("history", [])


def _load_pareto(path: Path) -> dict:
    """Carga pareto_metrics.json generado por el loop de desaprendizaje."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Gráfica 1 & 2: Eficacia y Utilidad vs pasos iterativos
# ---------------------------------------------------------------------------

def plot_sequential(domain: str = "literatura", n_batches: int = 5,
                    results_dir: Path = None, output_dir: Path = None):
    """
    Genera cuatro subgráficas mostrando la evolución de métricas a lo largo
    de los pasos de desaprendizaje:
      - ROUGE-1_forget   (↓ = mejor olvido)
      - PPL_forget       (↑ = mejor olvido)
      - ROUGE-1_retain   (↑ = modelo útil)
      - PPL_retain       (↓ = modelo útil)
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parents[1] / "results" / "runs"
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_r1f, ax_pplf), (ax_r1r, ax_pplr) = axes

    any_data = False
    for algo in ALGORITHMS:
        fpath = _result_path(results_dir, domain, algo, n_batches)
        if not fpath.exists():
            continue
        history = _load_history(fpath)
        if not history:
            continue
        any_data = True

        steps, r1f, pplf, r1r, pplr = [], [], [], [], []
        for rec in history:
            m = rec.get("metrics", {})
            eff = m.get("efficacy", {})
            util = m.get("utility", {})
            steps.append(rec["step"])
            r1f.append(eff.get("rouge1_forget", float("nan")))
            pplf.append(eff.get("ppl_forget", float("nan")))
            r1r.append(util.get("rouge1_retain", float("nan")))
            pplr.append(util.get("ppl_retain", float("nan")))

        kw = dict(color=COLORS.get(algo, "gray"), marker=MARKERS.get(algo, "o"),
                  linestyle=LINE_STYLES.get(algo, "-"), linewidth=2, markersize=7,
                  label=algo)

        ax_r1f.plot(steps, r1f, **kw)
        ax_pplf.plot(steps, pplf, **kw)
        ax_r1r.plot(steps, r1r, **kw)
        ax_pplr.plot(steps, pplr, **kw)

        # Annotate each point with its y-value
        for x, y in zip(steps, r1f):
            if not np.isnan(y):
                ax_r1f.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7, color=COLORS.get(algo, "gray"), path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        for x, y in zip(steps, pplf):
            if not np.isnan(y):
                ax_pplf.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7, color=COLORS.get(algo, "gray"), path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        for x, y in zip(steps, r1r):
            if not np.isnan(y):
                ax_r1r.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7, color=COLORS.get(algo, "gray"), path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        for x, y in zip(steps, pplr):
            if not np.isnan(y):
                ax_pplr.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7, color=COLORS.get(algo, "gray"), path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    if not any_data:
        print(f"[WARN] No se encontraron datos para dominio='{domain}', n_batches={n_batches}")
        plt.close(fig)
        return

    _style_ax(ax_r1f, "ROUGE-1 sobre Forget Set",
              "Paso Iterativo", "ROUGE-1 (fmeasure)", n_batches,
              note="↓ mejor olvido")
    _style_ax(ax_pplf, "Perplejidad sobre Forget Set",
              "Paso Iterativo", "PPL (log scale)", n_batches,
              note="↑ mejor olvido", log_y=True)
    _style_ax(ax_r1r, "ROUGE-1 sobre Retain Set",
              "Paso Iterativo", "ROUGE-1 (fmeasure)", n_batches,
              note="↑ mayor utilidad")
    _style_ax(ax_pplr, "Perplejidad sobre Retain Set",
              "Paso Iterativo", "PPL (log scale)", n_batches,
              note="↓ mayor utilidad", log_y=True)

    fig.suptitle(
        f"Desaprendizaje Secuencial — Dominio: {domain} | Lotes: {n_batches}",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = output_dir / f"{domain}_sequential_b{n_batches}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfica secuencial -> {out}")


# ---------------------------------------------------------------------------
# Gráfica 3: Frontera de Pareto FQ vs MU
# ---------------------------------------------------------------------------

def plot_pareto_frontier(domain: str = "literatura", n_batches_list: list = None,
                         checkpoints_dir: Path = None, output_dir: Path = None):
    """
    Scatter plot en el espacio (Model Utility, Forget Quality) para comparar
    algoritmos y niveles de granularidad del desaprendizaje secuencial.
    Genera dos versiones:
      1. Global (mostrando colapso catastrófico en x=0).
      2. Zoomed (filtrando modelos colapsados con MU < 0.1 para ver el detalle y trade-offs).
    """
    if n_batches_list is None:
        n_batches_list = [1, 3, 5]
    if checkpoints_dir is None:
        checkpoints_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in ["global", "zoomed"]:
        fig, ax = plt.subplots(figsize=(10, 7))
        all_points = []

        for algo in ALGORITHMS:
            for nb in n_batches_list:
                ppath = _pareto_path(checkpoints_dir, domain, algo, nb)
                if not ppath.exists():
                    continue
                pdata = _load_pareto(ppath)
                steps_data = pdata.get("steps", [])

                mus = []
                fqs = []
                for s in steps_data:
                    mu = float(s.get("model_utility", 0.0))
                    fq = float(s.get("forget_quality", 0.0))

                    # En el modo zoomed, filtramos los modelos que colapsaron completamente (MU < 0.1)
                    if mode == "zoomed" and mu < 0.1:
                        continue

                    mus.append(mu)
                    fqs.append(fq)
                    all_points.append((mu, fq))

                if not mus:
                    continue

                label = f"{algo} (b={nb})"
                alpha = 1.0  # Sin transparencia para colores fuertes
                size = 60 + 30 * n_batches_list.index(nb)
                ax.scatter(mus, fqs,
                           color=COLORS.get(algo, "gray"),
                           marker=MARKERS.get(algo, "o"),
                           s=size, alpha=alpha, label=label, zorder=3,
                           edgecolors="white", linewidths=0.5)

                # Conectar los puntos por paso con línea tenue
                if len(mus) > 1:
                    ax.plot(mus, fqs,
                            color=COLORS.get(algo, "gray"),
                            linestyle=LINE_STYLES.get(algo, "-"),
                            linewidth=1.2, alpha=0.85, zorder=2)

                # Anotar cada punto con su número de paso (P1, P2...) y sus coordenadas (MU, FQ)
                for idx, (mu, fq) in enumerate(zip(mus, fqs)):
                    step_num = steps_data[idx].get("step")
                    # En modo zoomed ponemos Paso + Coordenadas, en global solo el Paso para evitar sobrecarga
                    label = f"P{step_num}\n({mu:.3f}, {fq:.2f})" if mode == "zoomed" else f"P{step_num}"
                    ax.annotate(label, (mu, fq), textcoords="offset points", 
                                xytext=(5, 5), ha='left', va='bottom',
                                fontsize=6, color=COLORS.get(algo, "gray"), weight='bold', 
                                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        # Resaltar puntos Pareto-óptimos (máximo fq dado mu, sin ser dominados)
        if all_points:
            pareto_pts = _pareto_front(all_points)
            if pareto_pts:
                px, py = zip(*pareto_pts)
                ax.scatter(px, py, s=120, facecolors="none",
                           edgecolors="black", linewidths=2,
                           zorder=5, label="Pareto-óptimo")
                # Línea de frontera
                sorted_front = sorted(zip(px, py), key=lambda t: t[0])
                ax.step([p[0] for p in sorted_front], [p[1] for p in sorted_front],
                        where="post", color="black", linewidth=1.5, linestyle="--",
                        alpha=0.95, zorder=4)

        ax.set_xlabel("Model Utility (MU)  →  mayor es mejor", fontsize=12)
        ax.set_ylabel("Forget Quality (FQ) — KS p-value  →  mayor es mejor", fontsize=12)
        
        title_suffix = " (Global)" if mode == "global" else " (Zoomed — Modelos Estables)"
        ax.set_title(
            f"Frontera de Pareto: Forget Quality vs Model Utility{title_suffix}\n"
            f"Dominio: {domain}  |  Algoritmos: {', '.join(ALGORITHMS)}",
            fontsize=13, fontweight="bold",
        )
        
        if mode == "zoomed":
            # Leyenda afuera a la derecha para no tapar los puntos en el zoom
            ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.9, ncol=1)
        else:
            ax.legend(loc="lower right", framealpha=0.9, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.4)
        if mode == "zoomed" and all_points:
            all_fqs = [pt[1] for pt in all_points]
            max_fq = max(all_fqs) if all_fqs else 0.0
            ax.set_ylim(bottom=0, top=min(1.05, max(0.65, max_fq + 0.08)))
        else:
            ax.set_ylim(bottom=0, top=1.05)

        if mode == "global":
            ax.set_xlim(left=0)
            # Región ideal (alta FQ, alta MU)
            ax.axhspan(0.05, 1.05, xmin=0.7, alpha=0.04, color="green",
                       label="_nolegend_")
            ax.text(0.97, 0.97, "Zona\nideal", transform=ax.transAxes,
                    ha="right", va="top", color="green", fontsize=9, alpha=0.7)
        else:
            # En modo zoomed, dejamos que autoscale horizontalmente para ver los detalles
            # pero le damos un pequeño margen a los lados
            if all_points:
                all_mus = [pt[0] for pt in all_points]
                min_mu, max_mu = min(all_mus), max(all_mus)
                margin = (max_mu - min_mu) * 0.15 if max_mu > min_mu else 0.01
                ax.set_xlim(min_mu - margin, max_mu + margin)
                
                # Dibujamos zona ideal si cabe
                if max_mu > 0.2:
                    # En matplotlib, xmin/xmax para axhspan son coordenadas de datos en versiones modernas
                    # pero axvspan usa datos directamente, axhspan usa xmin/xmax en escala [0, 1] del eje x.
                    # Para evitar problemas de coordenadas relativas, usamos axvspan en el eje x o dejamos que autoscale sin sombreado verde
                    pass

        plt.tight_layout()
        suffix = "" if len(n_batches_list) > 1 else f"_b{n_batches_list[0]}"
        filename = f"{domain}_pareto_frontier{suffix}.png" if mode == "zoomed" else f"{domain}_pareto_frontier_global{suffix}.png"
        out = output_dir / filename
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Frontera de Pareto ({mode}) -> {out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, title, xlabel, ylabel, n_batches, note="", log_y=False):
    ax.set_title(f"{title}\n{note}", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, n_batches + 1))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    if log_y:
        ax.set_yscale("log")


def _pareto_front(points: list) -> list:
    """
    Devuelve los puntos Pareto-óptimos en el espacio (MU, FQ) bajo
    maximización de ambas coordenadas.
    Un punto p domina a q si p.mu >= q.mu y p.fq >= q.fq (con al menos uno estricto).
    """
    if not points:
        return []
    # Ordenar por MU descendente, luego por FQ descendente
    pts = sorted(set(points), key=lambda t: (-t[0], -t[1]))
    pareto = []
    max_fq = -float("inf")
    for mu, fq in pts:
        if fq >= max_fq:
            pareto.append((mu, fq))
            max_fq = fq
    return pareto


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Generando gráficas de resultados...")
    print("=" * 60)

    domains = ["literatura"]
    batch_configs = [1, 3, 5]

    for domain in domains:
        # Gráficas de evolución secuencial por granularidad
        for nb in batch_configs:
            try:
                plot_sequential(domain=domain, n_batches=nb)
            except Exception as e:
                print(f"[WARN] plot_sequential({domain}, b={nb}): {e}")

        # Frontera de Pareto consolidada (todos los n_batches juntos)
        try:
            plot_pareto_frontier(domain=domain, n_batches_list=batch_configs)
        except Exception as e:
            print(f"[WARN] plot_pareto_frontier({domain}): {e}")

        # Frontera de Pareto exclusiva para b=5
        try:
            plot_pareto_frontier(domain=domain, n_batches_list=[5])
        except Exception as e:
            print(f"[WARN] plot_pareto_frontier({domain}, b=5): {e}")

    print("\n[DONE] Todas las gráficas generadas.")


if __name__ == "__main__":
    main()