import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   # bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

ssu_lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    inference_mode=False,
    lora_dropout=0.01,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

# Define LoraConfig
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    inference_mode=False,
    lora_dropout=0.01,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

# Add any other configurations here
config = {
    "base_model_name":"mistralai/Mistral-7B-Instruct-v0.3",
    'use_quantization': False,
    'intervention':'unlearning_ga_none',
    'time_step_num':3,
    "random_loss_epsilon":0.5,
    'num_std_dev' : 1,
    'batch_size': 2,
    'lr': 1e-5,
    'max_unlearn_epochs': 1,
    "gradient_accumulation_steps":2,
    ####################### GA-based methods only ###########################
    'ga_factual_epsilon':0.5,
    ####################### NPO only ###########################
    'npo_beta': 0.4,
    ####################### SSU only ###########################
    "random_loss_pairs": 3,
}

initial_intervention = [
    "unlearning_gd_none",
]

available_intervention = [
    "sys_prompt-sys_none",
    "sys_prompt-sys_a",
    "unlearning_ga_none",
    "unlearning_npo_none",
    "unlearning_ga_idk",
    "unlearning_ga_mismatch",
    "unlearning_tv_none",
    "unlearning_tv_ssu",
    "mem_free_tokenized_consecutive",
]

oracle_model_for_npo = [
    "unlearning_npo_ref",
]

ablation_intervention = [
    "unlearning_tv_ssu_no_weight_saliency",
    "unlearning_tv_ssu_no_random_loss",
]
