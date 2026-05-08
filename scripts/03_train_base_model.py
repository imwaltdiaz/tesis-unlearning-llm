# scripts/03_train_base_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch
import os
import gc

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

def train_base_model(forget_books, retain_books, output_dir, domain="literatura"):
    """
    Entrena M0: modelo que conoce todos los libros
    Este es el modelo ANTES del unlearning
    """
    processed_dir = "data/processed/gutenberg"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    
    # LoRA config - eficiente para GPU limitada
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
    
    # Combina forget + retain para el entrenamiento base
    all_books = list(forget_books) + list(retain_books)
    data_files = [os.path.join(processed_dir, f"{b}.json") for b in all_books]
    missing = [p for p in data_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Faltan JSONs procesados. Ejecuta scripts/02_prepare_datasets.py primero. Ejemplo faltante: "
            + missing[0]
        )
    dataset = load_dataset("json", data_files=data_files, split="train")

    # Formato para SFTTrainer: columna única de texto
    def format_text(ex):
        inp = ex.get("input", "")
        out = ex.get("output", "")
        if inp and out and not inp.endswith((" ", "\n", "\t")):
            inp = inp + " "
        return {"text": inp + out}

    dataset = dataset.map(format_text, remove_columns=dataset.column_names)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=512,
        dataset_text_field="text",
    )
    
    trainer.train()

    # Guardar como modelo completo (merge LoRA) para que sea recargable
    out_path = f"{output_dir}/M0_{domain}"
    os.makedirs(out_path, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"M0 (merged) guardado en {out_path}")

    del trainer
    del model
    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path


if __name__ == "__main__":
    from pathlib import Path
    import importlib.util

    def _load_module(module_name: str, file_name: str):
        path = (Path(__file__).resolve().parent / file_name).as_posix()
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    ds = _load_module("_datasets", "02_prepare_datasets.py")
    train_base_model(ds.FORGET_BOOKS, ds.RETAIN_BOOKS, output_dir="checkpoints", domain="literatura")