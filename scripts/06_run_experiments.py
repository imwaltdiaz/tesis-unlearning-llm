# scripts/06_run_experiments.py
"""Script maestro que lanza todos los experimentos"""

from pathlib import Path
import importlib.util


def _load_module(module_name: str, file_name: str):
    """Load a Python module from a file path.

    Needed because these scripts use numeric prefixes (e.g. 04_*.py) which
    are not importable as regular Python modules.
    """
    path = (Path(__file__).resolve().parent / file_name).as_posix()
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


_datasets = _load_module("_datasets", "02_prepare_datasets.py")
_seq = _load_module("_seq", "04_sequential_unlearn.py")

FORGET_BOOKS = _datasets.FORGET_BOOKS
RETAIN_BOOKS = _datasets.RETAIN_BOOKS
prepare_batches = _seq.prepare_batches
sequential_unlearn_loop = _seq.sequential_unlearn_loop


def _result_path(domain: str, algorithm: str, n_batches: int) -> Path:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{domain}__{algorithm}__b{n_batches}.json"

ALGORITHMS = ["GA", "WGA", "NPO", "SimNPO"]
DOMAINS = ["literatura"]  # empieza solo con literatura, añade math e historia después
N_BATCHES = [1, 3, 5]    # 1 = baseline global, 3 y 5 = secuencial


def main():
    for domain in DOMAINS:
        for n_batches in N_BATCHES:
            for algorithm in ALGORITHMS:
                print(f"\n{'='*50}")
                print(f"Dominio: {domain} | Algo: {algorithm} | Lotes: {n_batches}")

                if algorithm == "SSU":
                    print("[SKIP] SSU no está integrado en este runner (usa SSU_Unlearn).")
                    continue

                result_file = _result_path(domain, algorithm, n_batches)
                if result_file.exists():
                    print(f"[SKIP] Ya existe resultado: {result_file}")
                    continue

                _, history = sequential_unlearn_loop(
                    M0_path=f"checkpoints/M0_{domain}",
                    forget_books_batched=prepare_batches(FORGET_BOOKS, n_batches),
                    retain_data=RETAIN_BOOKS,
                    algorithm=algorithm,
                    domain=domain,
                    n_batches=n_batches,
                )

                payload = {
                    "domain": domain,
                    "algorithm": algorithm,
                    "n_batches": n_batches,
                    "history": history,
                }
                result_file.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
                print(f"[OK] Guardado: {result_file}")


if __name__ == "__main__":
    main()