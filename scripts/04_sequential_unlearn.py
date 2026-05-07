# scripts/03_train_base_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch
import glob
import os


# -----------------------------
# Sequential Unlearning (thesis)
# -----------------------------


def load_model(model_path: str, torch_dtype=torch.bfloat16):
    """Carga modelo/tokenizer desde un checkpoint HF estándar.

    Nota: para evitar ambigüedad con LoRA (PEFT), este loader asume que
    `model_path` contiene un modelo completo (merged) compatible con
    `AutoModelForCausalLM.from_pretrained`.
    """
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
    """Carga el forget set y lo parte en `n_batches` segmentos contiguos.

    - n_batches=1 implementa la línea base global (forget set completo en un paso).
    - n_batches>1 implementa el desaprendizaje secuencial iterativo.
    """
    if n_batches < 1:
        raise ValueError("n_batches debe ser >= 1")

    forget_ds = _load_books_dataset(forget_books, processed_dir=processed_dir)
    n = len(forget_ds)
    if n == 0:
        raise ValueError("Forget dataset vacío")

    # Split contiguo: [0:cut1), [cut1:cut2), ...
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

        # Ensure EOS at the end of completion for stable loss
        full_text = prompt_for_full + completion
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.max_length)[
            "input_ids"
        ]
        full_ids = self.tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=self.max_length)[
            "input_ids"
        ]
        if len(full_ids) > 0 and full_ids[-1] != self.tokenizer.eos_token_id:
            # Respect max_length
            if len(full_ids) < self.max_length:
                full_ids = full_ids + [self.tokenizer.eos_token_id]
            else:
                full_ids[-1] = self.tokenizer.eos_token_id

        # Labels ignore the prompt tokens (supervised on completion only)
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        attn = [1] * len(full_ids)
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def sequential_unlearn_loop(
    *,
    M0_path: str,
    forget_books_batched,
    retain_data,
    algorithm: str,
    domain: str,
    n_batches: int,
    output_root: str = "checkpoints",
    processed_dir: str = "data/processed/gutenberg",
    max_length: int = 512,
    resume: bool = True,
    prefer_ref_on_gpu: bool = True,
):
    """Ejecuta desaprendizaje secuencial iterativo: M_n = U(M_{n-1}, L_n).

    - Para evitar OOM y garantizar la ecuación, cada paso carga M_{n-1} desde
      disco, entrena, evalúa y guarda M_n; luego libera explícitamente VRAM.
    """
    import sys
    import gc
    import json
    from pathlib import Path
    import importlib.util

    # Make OpenUnlearning importable without installation.
    ou_src = (Path(__file__).resolve().parents[1] / "open-unlearning" / "src").as_posix()
    if ou_src not in sys.path:
        sys.path.insert(0, ou_src)

    from omegaconf import OmegaConf
    from trainer import load_trainer
    from data.unlearn import ForgetRetainDataset
    from data.collators import DataCollatorForSupervisedDataset
    from transformers import set_seed
    from peft import get_peft_model

    # Load evaluate_full from scripts/05_evaluate.py (filename not importable as a module)
    eval_path = (Path(__file__).resolve().parent / "05_evaluate.py").as_posix()
    spec = importlib.util.spec_from_file_location("_ou_eval", eval_path)
    _ou_eval = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(_ou_eval)
    evaluate_full = _ou_eval.evaluate_full

    set_seed(42)

    # Perf knobs for RTX 6000 Ada
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    algo_map = {
        "GA": "GradAscent",
        "NPO": "NPO",
        "SimNPO": "SimNPO",
    }
    if algorithm not in algo_map:
        raise NotImplementedError(f"Algoritmo no soportado por este runner: {algorithm}")

    retain_hf = _load_books_dataset(retain_data, processed_dir=processed_dir)

    run_dir = os.path.join(output_root, domain, algorithm, f"batches_{n_batches}")
    os.makedirs(run_dir, exist_ok=True)

    history = []
    current_ckpt = M0_path

    # Simple auto-tuning based on GPU memory
    total_vram_gb = 0.0
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # Conservative defaults that still utilize a 48GB card well
    per_device_bs = 4 if total_vram_gb >= 40 else 1
    grad_accum = 4 if total_vram_gb >= 40 else 16
    num_workers = 8 if os.cpu_count() and os.cpu_count() >= 16 else 4

    for step_idx, forget_hf in enumerate(forget_books_batched, start=1):
        step_dir = os.path.join(run_dir, f"step_{step_idx}")
        os.makedirs(step_dir, exist_ok=True)

        step_model_ok = os.path.exists(os.path.join(step_dir, "config.json"))
        step_hist_path = os.path.join(step_dir, "history.json")

        # Resume logic: if step already produced a merged checkpoint, reuse it.
        if resume and step_model_ok:
            current_ckpt = step_dir
            if os.path.exists(step_hist_path):
                try:
                    with open(step_hist_path, "r", encoding="utf-8") as f:
                        history.append(json.load(f))
                except Exception:
                    history.append({"step": step_idx, "output_dir": step_dir, "metrics": None})
            else:
                history.append({"step": step_idx, "output_dir": step_dir, "metrics": None})
            continue

        # 1) Load M_{n-1}
        model, tokenizer = load_model(current_ckpt)
        model.config.use_cache = False

        # Enable gradient checkpointing for speed/memory tradeoff when available
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

        # 2) Apply LoRA (trainable adapters)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # 3) Build datasets (forget/retain) for OpenUnlearning
        forget_ds = _PromptCompletionTorchDataset(forget_hf, tokenizer, max_length=max_length)
        retain_ds = _PromptCompletionTorchDataset(retain_hf, tokenizer, max_length=max_length)
        unlearn_ds = ForgetRetainDataset(forget=forget_ds, retain=retain_ds, anchor="forget")
        collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        # 4) Trainer config
        trainer_cfg = {
            "handler": algo_map[algorithm],
            "args": {
                "output_dir": step_dir,
                "per_device_train_batch_size": per_device_bs,
                "gradient_accumulation_steps": grad_accum,
                "learning_rate": 2e-4,
                "num_train_epochs": 1,
                "logging_steps": 10,
                "save_strategy": "no",
                "report_to": "none",
                "bf16": True,
                "fp16": False,
                "remove_unused_columns": False,
                "dataloader_num_workers": num_workers,
                "dataloader_pin_memory": True,
                "gradient_checkpointing": True,
                "ddp_find_unused_parameters": False,
            },
            "method_args": {},
        }

        # 5) NPO: inject a CPU reference model loaded from checkpoint
        if algorithm == "NPO":
            ref_model = AutoModelForCausalLM.from_pretrained(current_ckpt, torch_dtype=torch.bfloat16)
            # With 48GB VRAM, keeping ref_model on GPU is usually safe and much faster.
            # If you want the most OOM-safe path, set prefer_ref_on_gpu=False.
            if torch.cuda.is_available() and prefer_ref_on_gpu and total_vram_gb >= 40:
                ref_model.to("cuda")
            else:
                ref_model.to("cpu")
            ref_model.eval()
            trainer_cfg["method_args"].update({"beta": 1.0, "ref_model": ref_model})
        else:
            ref_model = None

        trainer, _ = load_trainer(
            trainer_cfg=OmegaConf.create(trainer_cfg),
            model=model,
            train_dataset=unlearn_ds,
            eval_dataset=None,
            processing_class=tokenizer,
            data_collator=collator,
            evaluators=None,
            template_args=None,
        )

        # 6) Unlearn step: M_n = U(M_{n-1}, L_n)
        trainer.train()

        # 7) Eval on current forget batch
        eval_samples = [forget_hf[i] for i in range(min(10, len(forget_hf)))]
        metrics = evaluate_full(trainer.model, tokenizer, eval_samples, domain=domain, step=step_idx)
        step_record = {"step": step_idx, "metrics": metrics, "output_dir": step_dir}
        history.append(step_record)
        with open(step_hist_path, "w", encoding="utf-8") as f:
            json.dump(step_record, f, indent=2)

        # 8) Save as full (merged) model so next step can load cleanly
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(step_dir)
        tokenizer.save_pretrained(step_dir)
        current_ckpt = step_dir

        # Also write a run-level progress file for quick resumption/debug
        with open(os.path.join(run_dir, "history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(step_record) + "\n")

        # 9) Aggressive cleanup to prevent VRAM fragmentation/OOM
        del trainer
        del model
        del merged
        del forget_ds, retain_ds, unlearn_ds
        if ref_model is not None:
            del ref_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return current_ckpt, history

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

def load_and_format_dataset(data_dir="datos_procesados"):
    """Carga todos los JSONs y los formatea para el SFTTrainer"""
    # Busca todos los archivos JSON generados
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        raise ValueError(f"No se encontraron archivos JSON en {data_dir}")
    
    print(f"Cargando {len(json_files)} archivos para el modelo base (M0)...")
    dataset = load_dataset("json", data_files=json_files, split="train")
    
    # El SFTTrainer necesita una sola columna de texto continuo
    def format_text(example):
        return {"text": example["input"] + " " + example["output"]}
        
    dataset = dataset.map(format_text)
    return dataset

def train_base_model(output_dir="checkpoints", domain="literatura"):
    print("Iniciando entrenamiento de M0...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto" # ¡Esto volará en las GPUs del laboratorio!
    )
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Carga y formato de los datos reales
    dataset = load_and_format_dataset()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10, # Reducido para que veas el avance más rápido en el lab
        save_strategy="epoch",
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text", # Indicamos explícitamente qué columna usar
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    
    trainer.train()
    
    # Guardar el modelo final M0
    final_path = f"{output_dir}/M0_{domain}"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n¡Éxito! M0 guardado en {final_path}")
    
    return model

if __name__ == "__main__":
    train_base_model()