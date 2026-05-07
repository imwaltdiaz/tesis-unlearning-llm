import argparse
import os
import time

import torch
import math
from accelerate import Accelerator
from dotenv import load_dotenv
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from transformers import (
    get_scheduler,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import BitsAndBytesConfig
from tqdm import tqdm
import subprocess
import itertools
import utils
from config.fine_tune_config import bnb_config, lora_config, config, ssu_lora_config
from utils_data import create_copyrights_dataloader
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from task_vector import TaskVector
import torch.nn.functional as F
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def check_requires_grad(model):
    all_requires_grad_false = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} has requires_grad=True")
            all_requires_grad_false = False
        else:
            print(f"Parameter {name} has requires_grad=False")
    if all_requires_grad_false:
        print("All parameters have requires_grad set to False.")
    else:
        print("Some parameters have requires_grad set to True.")

def extract_model_name(input_string):
    # List of possible prefixes
    prefixes = ["meta-llama/", "mistralai/"]

    # Loop over prefixes to find a match
    for prefix in prefixes:
        if input_string.startswith(prefix):
            return input_string[len(prefix):]

    # Return the original string if no known prefix is found
    return input_string

def view_data_loader(data_loader):
    print(f"Number of batches in the data loader: {len(data_loader)}")
    for batch in data_loader:
        print("Batch:")
        print(batch.keys())
        input_ids = batch['input_ids']
        print("Input IDs:")
        # print(input_ids)
        print(input_ids.shape)

        attention_mask = batch['attention_mask']
        print("Attention Mask:")
        # print(attention_mask)
        print(attention_mask.shape)

        labels = batch['labels']
        print("Labels:")
        # print(labels)
        print(labels.shape)

        start_loc = batch['start_locs']
        print("Start Locations:")
        print(start_loc)
        break


def get_first_three_components(input_string):
    components = input_string.split('_')
    if len(components) > 3:
        return '_'.join(components[:3])
    else:
        return input_string

if __name__ == '__main__':
    torch.cuda.empty_cache()
    load_dotenv()

    parser = argparse.ArgumentParser(description='Fine-tune LLM on specific books')
    parser.add_argument('--model_dir', type=str, help='the directory of the models you saved locally')
    parser.add_argument('--data_dir', type=str, help='the directory of all the books you saved')
    parser.add_argument('--book_corpus_norm_data', type=str, help='the directory of the book corpus that does not have the books you want to unlearn')
    parser.add_argument('--Lora', action='store_true', default=False, help='Use Lora parameterization')
    parser.add_argument('--time_step_num', type=int, default=None, help='the time step number (optional, defaults to config value if not specified)')
    args = parser.parse_args()

    access_token = os.environ.get('HF_ACCESS_TOKEN')
    if not access_token:
        raise ValueError("Hugging Face access token not found. Please set the HF_ACCESS_TOKEN environment variable.")

    # accelerator = Accelerator()
    accelerator = Accelerator(device_placement=True, mixed_precision='fp16', split_batches=True)
    device = accelerator.device

    base_model_name = config['base_model_name']
    print("base_model_name is", base_model_name)
    print("config['intervention'] is", config['intervention'])
    print("config[use_quantization] is", config["use_quantization"])

    time_step_num = args.time_step_num if args.time_step_num is not None else config["time_step_num"]
    if base_model_name in ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
        print(f"Loading model {base_model_name} at time step {time_step_num}")
        modified_base_model_name = extract_model_name(base_model_name)
        print("Modified base model name is", modified_base_model_name)
        intervention = config["intervention"]
        intervention_choice = config["intervention"].split('_')
        if time_step_num == 1:
            if intervention_choice[1] == "gd":
                if config["use_quantization"]:
                    model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
                                                                 device_map="auto", token=access_token,
                                                                 cache_dir=args.model_dir)
                else:
                    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16,
                                                                 device_map="auto", token=access_token,
                                                                 cache_dir=args.model_dir)
            else:
                modified_base_model_name_edited = f"Mistral-7B-Instruct-v0.3_time_step_10_intervention_unlearning_gd_none"
                model_checkpoint_path = os.path.join(args.model_dir, modified_base_model_name_edited)
                print("Loading Model checkpoint path: ", model_checkpoint_path)
                if config["use_quantization"]:
                    model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, quantization_config=bnb_config,
                                                                 device_map="auto", token=access_token,
                                                                 cache_dir=args.model_dir)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                                 device_map="auto",
                                                                 token=access_token,
                                                                 torch_dtype=torch.float16,
                                                                 cache_dir=args.model_dir)
        elif time_step_num > 1:
            if intervention != "unlearning_tv_ssu_no_weight_saliency" and intervention != "unlearning_tv_ssu_no_random_loss":
                print("Not running ablation studies")
                intervention_to_load = get_first_three_components(intervention)
            else:
                intervention_to_load = intervention
            modified_base_model_name_modified = f"{modified_base_model_name}_time_step_{time_step_num-1}_intervention_{intervention_to_load}"
            model_checkpoint_path = os.path.join(args.model_dir, modified_base_model_name_modified)
            print("Provided intervention is", intervention)
            print("Loading model from checkpoint", model_checkpoint_path)
            if config["use_quantization"]:
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, quantization_config=bnb_config,
                                                             device_map="auto", token=access_token,
                                                             cache_dir=args.model_dir)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, torch_dtype=torch.float16,
                                                             device_map="auto", token=access_token,
                                                             cache_dir=args.model_dir)
        else:
            raise ValueError("The time step number is not valid. It has to be at least one")
        # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    else:
        raise ValueError("The model type is not supported.")

    intervention_choice = config["intervention"].split('_')
    assert intervention_choice[0] == "unlearning"
    assert intervention_choice[1] in ["ga", "tv", "npo", "gd"]
    unlearning_method = intervention_choice[1]
    unlearning_choice = intervention_choice[2]

    if args.Lora:
        if unlearning_method == "tv" and unlearning_choice == "ssu":
            print("TV SSU Lora Config selected")
            model = get_peft_model(model, ssu_lora_config)
            target_modules = list(ssu_lora_config.target_modules)
            print(f"Target Modules: {target_modules}")
        else:
            print("General Lora Config Selected")
            model = get_peft_model(model, lora_config)
            target_modules = list(lora_config.target_modules)
            print(f"Target Modules: {target_modules}")
        print("LoRA configuration added.")
        model.print_trainable_parameters()
    else:
        print("No Lora Used.")


    book_dir_file_path = f'data_csv_single/time_step_{time_step_num}/time_step_{time_step_num}_train_dataset_unlearn.json'
    data_path = os.path.join(args.data_dir, book_dir_file_path)
    print("data_path used for unlearning is", data_path)
    _, unlearn_data_loader, unlearn_book_all_ans = create_copyrights_dataloader(file_path=data_path,
                                                                                tokenizer=tokenizer,
                                                                                batch_size=config['batch_size'])

    print(f"Number of batches in the unlearn data loader: {len(unlearn_data_loader)}")

    # print("View data loader")
    # print(view_data_loader(unlearn_data_loader))
    # print("\n")

    book_corpus_norm_data_path = os.path.join(args.data_dir, args.book_corpus_norm_data)

    print(f"book_corpus_norm_data_path : {book_corpus_norm_data_path}")
    _, book_corpus_norm_data_loader, book_corpus_norm_all_ans = create_copyrights_dataloader(
        file_path=book_corpus_norm_data_path,
        tokenizer=tokenizer,
        batch_size=config['batch_size'])

    print(f"Number of batches in the book_corpus_norm_data loader: {len(book_corpus_norm_data_loader)}")

    num_training_steps=len(unlearn_data_loader)*config['max_unlearn_epochs']
    print(f"Number of training steps: {num_training_steps}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    if unlearning_method == "npo" and unlearning_choice != "ref":
        modified_base_model_name = extract_model_name(base_model_name)
        modified_base_model_name_modified = f"{modified_base_model_name}_time_step_{time_step_num}_intervention_unlearning_npo_ref"
        model_checkpoint_path = os.path.join(args.model_dir, modified_base_model_name_modified)
        print("Loading oracle model from", model_checkpoint_path)
        oracle_model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, torch_dtype=torch.float16,
                                                             device_map="auto", token=access_token,
                                                             cache_dir=args.model_dir)
        oracle_model.eval()

    if unlearning_method != "npo" or unlearning_choice == "ref":
        (
            model,
            optimizer,
            unlearn_data_loader,
            book_corpus_norm_data_loader,
            lr_scheduler,
        ) = accelerator.prepare(
            model, optimizer, unlearn_data_loader, book_corpus_norm_data_loader, lr_scheduler
        )
    else:
        (
            model,
            oracle_model,
            optimizer,
            unlearn_data_loader,
            book_corpus_norm_data_loader,
            lr_scheduler,
        ) = accelerator.prepare(
            model, oracle_model, optimizer, unlearn_data_loader, book_corpus_norm_data_loader, lr_scheduler
        )

    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    optimizer.zero_grad()
    model.train()

    pad_token_id = tokenizer.pad_token_id
    print(f"Pad token ID: {pad_token_id}")

    output_file_name = f'{modified_base_model_name}_time_step_{time_step_num}_intervention_{config["intervention"]}'
    output_dir = os.path.join(args.model_dir, output_file_name)
    print(f"The output dir is {output_dir}")

    # Train the model
    start_time = time.time()

    idk_normal_answer = ["I'm sorry, I don't have enough information to answer that question."]*10
    epsilon_rl = config["random_loss_epsilon"]

    unlearn_data_loader_cycle = itertools.cycle(unlearn_data_loader)
    book_corpus_norm_data_loader_cycle = itertools.cycle(book_corpus_norm_data_loader)
    for idx in tqdm(range(num_training_steps), total=num_training_steps, desc="Training Progress"):
        batch_norm = next(book_corpus_norm_data_loader_cycle)
        unlearn_batch_custom = next(unlearn_data_loader_cycle)
        if unlearning_method == 'gd':
            gradient_loss_custom = utils.get_answer_loss(operation="gd", model=model, batch=unlearn_batch_custom,
                                                         device=accelerator.device,
                                                         pad_token_id=pad_token_id)

            random_loss = 0
            gradient_loss_factual = 0
        elif unlearning_method == "ga":
            gradient_loss_custom = utils.get_answer_loss(operation="ga", model=model, batch=unlearn_batch_custom,
                                                         device=accelerator.device,
                                                         pad_token_id=pad_token_id)

            if unlearning_choice=="idk":
                random_loss = utils.get_rand_ans_loss(
                    bad_batch=unlearn_batch_custom,
                    tokenizer=tokenizer,
                    normal_ans=idk_normal_answer,
                    model=model,
                    pad_token_id=pad_token_id,
                    K=1,
                    device=accelerator.device,
                )
            elif unlearning_choice=="mismatch":
                random_loss = utils.get_rand_ans_loss(
                    bad_batch=unlearn_batch_custom,
                    tokenizer=tokenizer,
                    normal_ans=book_corpus_norm_all_ans,
                    model=model,
                    pad_token_id=pad_token_id,
                    K=3,
                    device=accelerator.device,
                )
            elif unlearning_choice=="none":
                random_loss = 0
            else:
                raise ValueError(f"Unlearning choice is not supported for the unlearning method {unlearning_method}.")
        elif unlearning_method == "tv":
            gradient_loss_custom = utils.get_answer_loss(operation="gd",
                                                         model=model,
                                                         batch=unlearn_batch_custom,
                                                         device=accelerator.device,
                                                         pad_token_id=pad_token_id)

            if unlearning_choice=="ssu" and config["intervention"] != "unlearning_tv_ssu_no_random_loss":
                random_loss = utils.get_rand_ans_loss(
                    bad_batch=unlearn_batch_custom,
                    tokenizer=tokenizer,
                    normal_ans=unlearn_book_all_ans,
                    model=model,
                    pad_token_id=pad_token_id,
                    K=config["random_loss_pairs"],
                    device=accelerator.device,
                )
            elif unlearning_choice=='none' or config["intervention"] == "unlearning_tv_ssu_no_random_loss":
                random_loss=0
            else:
                raise ValueError(f"Unlearning choice is not supported for the unlearning method {unlearning_method}.")
        elif unlearning_method == "npo" and unlearning_choice != "ref":
            print("Calculating NPO loss")

            # Calculate the NPO loss using the new method
            gradient_loss_custom = utils.calculate_npo_loss(
                unlearn_batch_custom=unlearn_batch_custom,
                model=model,
                oracle_model=oracle_model,
                pad_token_id=pad_token_id,
                beta=config['npo_beta'],
                device=accelerator.device,
            )

            if unlearning_choice=="none":
                random_loss = 0
        elif unlearning_method == "npo" and unlearning_choice == "ref":
            gradient_loss_custom = utils.get_answer_loss(operation="gd", model=model, batch=unlearn_batch_custom,
                                                         device=accelerator.device,
                                                         pad_token_id=pad_token_id)
            gradient_loss_custom += utils.get_answer_loss(operation="gd", model=model, batch=batch_norm,
                                  device=accelerator.device,
                                  pad_token_id=pad_token_id)
            random_loss = 0
        else:
            raise ValueError(f"Unlearning method {unlearning_method} is not supported.")
        gradient_loss_custom = gradient_loss_custom.to(accelerator.device)
        if random_loss != 0:
            random_loss = random_loss.to(accelerator.device)

        if unlearning_method=="ga" and unlearning_choice!="none":

            gradient_loss_factual = utils.get_answer_loss(operation="gd", model=model, batch=batch_norm,
                                  device=accelerator.device,
                                  pad_token_id=pad_token_id)

            gradient_loss_factual = gradient_loss_factual.to(accelerator.device)

            print(f"gradient_loss_factual (scaled): {gradient_loss_factual:.4f}")

            if math.isnan(gradient_loss_factual):
                print("Gradient Loss Factual loss is NaN")
                gradient_loss_factual = 0
        else:
            gradient_loss_factual = 0

        if config["intervention"] == "unlearning_tv_ssu_no_random_loss":
            epsilon_rl = 0

        if gradient_loss_factual != 0 and random_loss != 0:
            print("Factual loss and random loss are not zero")
            gradient_loss =  gradient_loss_custom + epsilon_rl * random_loss + config['ga_factual_epsilon']*gradient_loss_factual
        elif random_loss != 0:
            print("Random loss (without epsilon term) is not zero")
            gradient_loss = gradient_loss_custom + epsilon_rl * random_loss
        else:
            print("Factual and Random loss are zero")
            gradient_loss = gradient_loss_custom

        avg_loss = gradient_loss.item()

        print(
            f"Step {idx}/{num_training_steps} - Average Loss: {avg_loss:.4f}, Gradient Loss Custom: {config['ga_factual_epsilon'] * gradient_loss_custom:.4f}, " 
            f"Random Loss: { epsilon_rl * random_loss:.4f}, Gradient Loss Factual: {gradient_loss_factual:.4f}")

        if unlearning_method=="tv" and unlearning_choice=="ssu" and "no_weight_saliency" not in config["intervention"]:
            # ======================================== Weight Saliency ========================================
            num_std_dev = config["num_std_dev"]
            accelerator.backward(gradient_loss, retain_graph=True)
            saliency_masks = utils.compute_saliency_map(model, device=accelerator.device, target_modules=target_modules, num_std_dev=num_std_dev)
            torch.cuda.empty_cache()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # print param name, and the corresponding requires_grad value
                    # print(f"Parameter {name} has requires_grad={param.requires_grad} with value {param.grad}")
                    param.grad.data = param.grad.data * saliency_masks[name].to(param.grad.device)
            torch.cuda.empty_cache()
            accelerator.backward(gradient_loss)
            torch.cuda.empty_cache()
            # ================================================================================================
        else:
            accelerator.backward(gradient_loss)

        if (idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    if args.Lora:
        model = model.merge_and_unload()
        print("Model merged and unloaded.")
    else:
        print("No Lora Used.")

    print("Unlearning finished")
    end_time = time.time()
    print(f"Total time taken to fine-tune: {end_time - start_time} seconds")

    torch.cuda.empty_cache()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if unlearning_method=="tv":
        print("Start applying task vector to the model.")
        base_model_name = config['base_model_name']
        # pre_trained_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
        #                                                          device_map="auto", token=access_token,
        #                                                          cache_dir=args.model_dir)
        modified_base_model_name = extract_model_name(base_model_name)
        # if time_step_num == 1:
        #     if config["use_quantization"]:
        #         pre_trained_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
        #                                                                  device_map="auto", token=access_token,
        #                                                                  cache_dir=args.model_dir)
        #     else:
        #         pre_trained_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16,
        #                                                                  device_map="auto", token=access_token,
        #                                                                  cache_dir=args.model_dir)
        if time_step_num == 1:
            modified_base_model_name_edited = f"Mistral-7B-Instruct-v0.3_time_step_10_intervention_unlearning_gd_none"
            model_checkpoint_path = os.path.join(args.model_dir, modified_base_model_name_edited)
            print("Loading Model checkpoint path: ", model_checkpoint_path)
            if config["use_quantization"]:
                pre_trained_model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, quantization_config=bnb_config,
                                                             device_map="auto", token=access_token,
                                                             cache_dir=args.model_dir)
            else:
                pre_trained_model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                             device_map="auto",
                                                             token=access_token,
                                                             torch_dtype=torch.float16,
                                                             cache_dir=args.model_dir)
        elif time_step_num > 1:
            intervention = config["intervention"]
            if intervention != "unlearning_tv_ssu_no_weight_saliency" and intervention != "unlearning_tv_ssu_no_random_loss":
                print("Not running ablation studies")
                intervention_to_load = get_first_three_components(intervention)
            else:
                intervention_to_load = intervention
            modified_base_model_name_modified = f"{modified_base_model_name}_time_step_{time_step_num - 1}_intervention_{intervention_to_load}"
            model_checkpoint_path = os.path.join(args.model_dir, modified_base_model_name_modified)
            print("Provided intervention is", intervention)
            print("Loading model from checkpoint", model_checkpoint_path)
            if config["use_quantization"]:
                pre_trained_model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, quantization_config=bnb_config,
                                                                         device_map="auto", token=access_token,
                                                                         cache_dir=args.model_dir)
            else:
                pre_trained_model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, torch_dtype=torch.float16,
                                                                         device_map="auto", token=access_token,
                                                                         cache_dir=args.model_dir)
        else:
            raise ValueError("The time step number is not valid. It has to be at least one")

        task_vector = TaskVector(pretrained_checkpoint=pre_trained_model, finetuned_checkpoint=model)
        neg_task_vector = -task_vector

        new_unlearned_model = neg_task_vector.apply_to(pretrained_model=pre_trained_model)
        print("Task vector applied to the model.")
        print(new_unlearned_model)
        model = new_unlearned_model
    print("The model will be saved to", output_dir)
    model.save_pretrained(output_dir, from_pt=True)
    print("Model saved.")
