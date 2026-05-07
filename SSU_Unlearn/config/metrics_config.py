import torch
from transformers import BitsAndBytesConfig

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_8bit_use_double_quant=True,
#     bnb_8bit_quant_type="nf4",
#     bnb_8bit_compute_dtype=torch.float16,
# )

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   # bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

memFree_Prompt_Config = {
    # "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "model_name":"mistralai/Mistral-7B-Instruct-v0.3",
    "is_instruct_model" : True, # Whether the model is an instruction following or a chat model
    "use_quantization" : False, # Don't use quantization
    "train_or_test":"train",
    "model_dir": "/cache", # The directory where the model is saved
    "datatype": "gutenberg_books",
    "n" : 6, # The choice of n-grams for MemFree decoding
    "num_tests" : 200, # The number of QA pairs to run for evaluation
    "intervention":"unlearning_ga_none",
    'eval_general':False,

    # "intervention": "sys_prompt-sys_a",
    "acs_threshold": 50, # for non-consecutive case
    "no_context": False,
    "no_overwrite":False,
}

available_intervention = [
    "sys_prompt-sys_none",
    "sys_prompt-sys_a",
    "sys_prompt-dbrx",
    "unlearning_ga_none",
    "unlearning_npo_none",
    "unlearning_ga_idk",
    "unlearning_ga_mismatch",
    "unlearning_tv_none",
    "unlearning_tv_ssu",
    "mem_free_tokenized_consecutive",
]

ablation_intervention = [
    "unlearning_tv_ssu_no_weight_saliency",
    "unlearning_tv_ssu_no_random_loss",
]
