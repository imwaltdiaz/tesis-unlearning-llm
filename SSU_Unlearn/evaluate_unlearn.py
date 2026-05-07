import argparse
import json
import os
import pandas as pd
import numpy as np
import string
import subprocess
import evaluate
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import tqdm
import timeit
import random
import utils
from config.metrics_config import bnb_config, config
from CoTaEval.lib.prompt_utils import apply_prompt_template
import dataportraits
import re
from CoTaEval.lib.decoding_intervention import DataPortraitsLogitsProcessor, TopKPerturbationLogitsProcessor, DataPortraitsSkipLogitsProcessor
from config.metrics_config import memFree_Prompt_Config
import CoTaEval.lib.utils as utils
from CoTaEval.process import add_metrics

# def extract_model_name(input_string):
#     target_prefix = "meta-llama/"
#     if input_string.startswith(target_prefix):
#         return input_string[len(target_prefix):]
#     return input_string

def extract_model_name(input_string):
    # List of possible prefixes
    prefixes = ["meta-llama/", "mistralai/"]

    # Loop over prefixes to find a match
    for prefix in prefixes:
        if input_string.startswith(prefix):
            return input_string[len(prefix):]

    # Return the original string if no known prefix is found
    return input_string

def extract_last_x_tokens(text, x):
    # Tokenize the string
    tokens = text.split()  # You can modify the delimiter in split() if needed

    # Extract the last x tokens
    last_x_tokens = tokens[-x:] if x <= len(tokens) else tokens

    # Join them back into a string if needed
    return ' '.join(last_x_tokens)

def get_substring_after(A, B):
    index = A.find(B)
    if index == -1:
        return False, "String B not found in String A."
    # Calculate the end index of B in A
    end_index = index + len(B)
    # Return the substring from the end of B to the end of A
    return True, A[end_index:]

def mmlu_only(n, base_model_name,  model_dir, args):
    torch.cuda.empty_cache()
    load_dotenv()
    random_seed = 42

    intervention = memFree_Prompt_Config["intervention"]
    bf_is_tokenized = "tokenized" in intervention
    print("Intervention is: ", intervention)

    time_step_num = args.time_step_num
    intervention = memFree_Prompt_Config["intervention"]
    if "tokenized" in intervention:
        bloom_filter = f'gutenberg_books_time_step_{time_step_num}_tokenized.{6 * n}-{6 * n}.bf'
    else:
        bloom_filter = f'gutenberg_books_time_step_{time_step_num}.{n}-{n}.bf'

    intervention_choice = intervention.split('_')
    print("Test Bloom Filter name is:", bloom_filter)
    access_token = os.environ.get('HF_ACCESS_TOKEN')
    print("Model directory is: ", model_dir)
    if not access_token:
        raise ValueError("Hugging Face access token not found. Please set the HF_ACCESS_TOKEN environment variable.")
    print("intervention_choice is: ", intervention_choice)
    if base_model_name in ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B",
                           "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
        print(f"Loading model {base_model_name}")

        if intervention_choice[0] == 'unlearning':
            print("Running machine unlearning interventions")
            modified_base_model_name = extract_model_name(base_model_name)
            modified_base_model_name = f"{modified_base_model_name}_time_step_{time_step_num}_intervention_{intervention}"
            model_checkpoint_path = os.path.join(args.model_dir, modified_base_model_name)
            print("Model checkpoint path: ", model_checkpoint_path)
            if memFree_Prompt_Config["use_quantization"]:
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                             quantization_config=bnb_config,
                                                             device_map="auto",
                                                             token=access_token,
                                                             cache_dir=args.model_dir)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                             device_map="auto",
                                                             token=access_token,
                                                             torch_dtype=torch.float16,
                                                             cache_dir=args.model_dir)
        else:
            print("Running non-machine unlearning interventions")
            print("Base model name is: ", base_model_name)
            modified_base_model_name = f"Mistral-7B-Instruct-v0.3_time_step_10_intervention_unlearning_gd_none"
            model_checkpoint_path = os.path.join(args.model_dir, modified_base_model_name)
            print("Loading Model checkpoint path: ", model_checkpoint_path)
            if memFree_Prompt_Config["use_quantization"]:
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                             quantization_config=bnb_config,
                                                             device_map="auto",
                                                             token=access_token,
                                                             cache_dir=model_dir)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                             device_map="auto",
                                                             token=access_token,
                                                             torch_dtype=torch.float16,
                                                             cache_dir=model_dir)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)
        if "mistral" in base_model_name or "Mistral-7B" in base_model_name:
            tokenizer.pad_token = "<pad>"
        else:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
    else:
        raise ValueError(f"Model {base_model_name} not supported.")

    mmlu_score = eval_mmlu(args,model=model, tokenizer=tokenizer, bloom_filter=bloom_filter)
    model_name_for_save = extract_model_name(memFree_Prompt_Config["model_name"])
    save_path = f"SSU_Unlearn/res/mmlu_res/log_{model_name_for_save}_{intervention}_time_step_{time_step_num}.txt"
    print("save path is", save_path)
    return mmlu_score

def main(n, file_path, base_model_name, use_fine_tuned_model, fine_tuned_model_name, fine_tuned_filename, model_dir, args):
    torch.cuda.empty_cache()
    load_dotenv()
    random_seed = 42

    intervention = memFree_Prompt_Config["intervention"]
    print("Intervention is: ", intervention)
    num_tests = memFree_Prompt_Config["num_tests"]

    bf_is_tokenized = "tokenized" in intervention

    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            testing_chunks = pd.DataFrame(data)
    else:
        assert file_path.endswith('.csv')
        testing_chunks = pd.read_csv(file_path)

    print("length of original total testing chunks", len(testing_chunks))
    if not args.use_all:
        print("Not using all data")
        shuffled_testing_chunks = testing_chunks.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        testing_chunks = shuffled_testing_chunks.head(num_tests)
        print("length of modified total testing chunks", len(testing_chunks))

    time_step_num = args.time_step_num
    intervention = memFree_Prompt_Config["intervention"]
    if "tokenized" in intervention:
        bloom_filter = f'gutenberg_books_time_step_{time_step_num}_tokenized.{6 * n}-{6 * n}.bf'
    else:
        bloom_filter = f'gutenberg_books_time_step_{time_step_num}.{n}-{n}.bf'

    intervention_choice = intervention.split('_')
    print("Test Bloom Filter name is:", bloom_filter)
    access_token = os.environ.get('HF_ACCESS_TOKEN')
    print("Model directory is: ", model_dir)
    if not access_token:
        raise ValueError("Hugging Face access token not found. Please set the HF_ACCESS_TOKEN environment variable.")
    print("intervention_choice is: ", intervention_choice)
    if base_model_name in ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
        print(f"Loading model {base_model_name}")

        if intervention_choice[0] == 'unlearning':
            print("Running machine unlearning interventions")
            modified_base_model_name = extract_model_name(base_model_name)
            modified_base_model_name = f"{modified_base_model_name}_time_step_{time_step_num}_intervention_{intervention}"
            model_checkpoint_path = os.path.join(args.model_dir, modified_base_model_name)
            print("Model checkpoint path: ", model_checkpoint_path)
            if memFree_Prompt_Config["use_quantization"]:
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                             quantization_config=bnb_config,
                                                             device_map="auto",
                                                             token=access_token,
                                                             cache_dir=args.model_dir)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                             device_map="auto",
                                                             token=access_token,
                                                             torch_dtype=torch.float16,
                                                             cache_dir=args.model_dir)
        else:
            print("Running non-machine unlearning interventions")
            print("Base model name is: ", base_model_name)
            modified_base_model_name = f"Mistral-7B-Instruct-v0.3_time_step_10_intervention_unlearning_gd_none"
            model_checkpoint_path = os.path.join(args.model_dir, modified_base_model_name)
            print("Loading Model checkpoint path: ", model_checkpoint_path)
            if memFree_Prompt_Config["use_quantization"]:
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                             quantization_config=bnb_config,
                                                             device_map="auto",
                                                             token=access_token,
                                                             cache_dir=model_dir)
            else:

                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                             device_map="auto",
                                                             token=access_token,
                                                             torch_dtype=torch.float16,
                                                             cache_dir=model_dir)
                # model = AutoModelForCausalLM.from_pretrained(base_model_name,
                #                                              device_map="auto",
                #                                              token=access_token,
                #                                              torch_dtype=torch.float16,
                #                                              cache_dir=model_dir)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)
        if "mistral" in base_model_name or "Mistral-7B" in base_model_name:
            tokenizer.pad_token = "<pad>"
        else:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"

        if use_fine_tuned_model:
            assert fine_tuned_model_name is not None
            assert fine_tuned_filename is not None
            print(f"Loading fine-tuned model {fine_tuned_model_name} with filename {fine_tuned_filename}")
            fine_tuned_model_path = os.path.join(model_dir, fine_tuned_model_name)

            print("Fine-tuned model path: ", fine_tuned_model_path)
            _ = utils.load_checkpoint(model, checkpoint_dir=fine_tuned_model_path, filename=fine_tuned_filename)
    else:
        raise ValueError(f"Model {base_model_name} not supported.")

    print(model)
    print("Model loaded successfully \n")
    prior_processor = model._get_logits_processor
    model.generation_config.context_aware_decoding_alpha = None
    model.generation_config.mem_free_new = False

    output_list, prompt_list, gt_list, inference_time_list = [], [], [], []
    print("Testing chunks length: ", len(testing_chunks))
    for i, (prompt, gt) in tqdm.tqdm( enumerate(zip(testing_chunks['question'], testing_chunks['answer'])), total=len(testing_chunks)):
        print(f"Testing chunk {i + 1} / {len(testing_chunks)}")
        if 'mem_free' in intervention:
            choice = intervention.split('-')[-1]
            print("The choice is ", choice, "with intervention", intervention)
            print(f"Prompt (question): {prompt} \n")
            if no_context:
                context = ""
            else:
                context = f"Context: {prompt + ' ' + gt}\n"
            if "llama3" in base_model_name or "Llama-3" in base_model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context, model="llama3")[0]
            elif "mistral" in base_model_name or "Mistral-7B" in base_model_name:
                prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context, model="mistral")[0]
            else:
                prompt = apply_prompt_template(prompt_template_style=choice, dataset=[prompt], context=context)[0]
            matches = re.findall(r'\d+', bloom_filter)
            if len(matches) > 1:
                n = matches[1]
            else:
                n = matches[0]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            portrait = dataportraits.RedisBFSketch('localhost', 6379, bloom_filter, int(n)) # n would be 36
            print("Successfully loaded testing chunks from", bloom_filter)
            if choice == 'consecutive':
                if bf_is_tokenized:
                    width = 2 * int(n) - 6
                else:
                    width = 2 * int(n) - 1

                def new_logits_processor(*args, **kwargs):
                    prior = prior_processor(*args, **kwargs)
                    prior.append(
                        DataPortraitsLogitsProcessor(prompt=prompt, width=width, tokenizer=tokenizer, portrait=portrait, tokenized_prompt=inputs, bf_is_tokenized=bf_is_tokenized,
                                                      n=int(n), consecutive=True))
                    return prior
            else:
                width = 3 * int(n)

                def new_logits_processor(*args, **kwargs):
                    prior = prior_processor(*args, **kwargs)
                    prior.append(
                        DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized,
                                                     tokenized_prompt=inputs, n=int(n), consecutive=False,
                                                     acs_threshold=acs_threshold))
                    return prior
            model._get_logits_processor = new_logits_processor
            time_start = timeit.default_timer()
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=max_completion_tokens, do_sample=False,  min_new_tokens=min_new_tokens,
                                          num_return_sequences=1, pad_token_id=tokenizer.pad_token_id,
                                          attention_mask=inputs.attention_mask)
            time_end = timeit.default_timer()
        elif 'sys_prompt' in intervention:
            print("Intervention is", intervention)
            if no_context:
                context = ""
            else:
                context = f"Context: {prompt + ' ' + gt}\n"
            system_prompt_choice = intervention.split('-')[-1]
            print("The system prompt choice is: ", system_prompt_choice)
            if 'llama2' in base_model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], context=context)[0]
            elif 'dbrx' in base_model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], context=context, model='dbrx')[0]
            elif 'llama3' in base_model_name or "Llama-3" in base_model_name:
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], context=context, model='llama3')[0]
            elif 'mistral' in base_model_name or "Mistral-7B" in base_model_name:
                # source: https://web.archive.org/web/20231030013339/https://docs.mistral.ai/usage/guardrailing/#appendix
                prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], context=context, model='mistral')[0]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            time_start = timeit.default_timer()
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=max_completion_tokens, do_sample=True, min_new_tokens=min_new_tokens,
                                          temperature=0.4, top_p=0.8, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id,
                                          attention_mask=inputs.attention_mask)  # The only difference is do_sample
            # generate_ids = model.generate(inputs.input_ids, max_new_tokens=200, do_sample=False,
            #                               min_new_tokens=min_new_tokens,
            #                               num_return_sequences=1,
            #                               pad_token_id=tokenizer.pad_token_id,
            #                               attention_mask=inputs.attention_mask)  # The only difference is do_sample
            time_end = timeit.default_timer()
        else:
            if no_context:
                context = ""
            else:
                context = f"Context: {prompt + ' ' + gt}\n"
            if any(element in base_model_name for element in ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf', 'dbrx',
                                                              'llama3-8b-chat-hf',
                                                              "meta-llama/Meta-Llama-3-8B",
                                                              "meta-llama/Meta-Llama-3.1-8B",
                                                              "meta-llama/Meta-Llama-3.1-8B-Instruct",
                                                              "mistralai/Mistral-7B-Instruct-v0.3",
                                                              "meta-llama/Llama-3.2-3B-Instruct"]):
                if "llama2" in base_model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context)[0]
                elif "dbrx" in base_model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context, model="dbrx")[0]
                elif 'llama3' in base_model_name or "Llama-3" in base_model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], context=context, model="llama3")[0]
                elif "mistral" in base_model_name or "Mistral-7B" in base_model_name:
                    prompt = apply_prompt_template(prompt_template_style='sys_a', dataset=[prompt], context=context, model="mistral")[0]
                    print("Prompt after applying prompt template: ", prompt)
            elif "llama2-7b-hf" in base_model_name: # For base model case and we only evaluate non-context situation.
                assert no_context==True
            else:
                raise NotImplementedError

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            time_start = timeit.default_timer()
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=max_completion_tokens, do_sample=True,
                                          min_new_tokens=min_new_tokens,
                                          temperature=0.4, top_p=0.8, num_return_sequences=1,
                                          pad_token_id=tokenizer.pad_token_id,
                                          attention_mask=inputs.attention_mask)  # The only difference is do_sample
            time_end = timeit.default_timer()


        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if "mistral" in base_model_name or "Mistral-7B" in base_model_name:
            prompt = prompt.replace("[INST]", "").replace("[/INST]", "")
            prompt_length = len(prompt)
            # print("Prompt length is: ", prompt_length)
            cleaned_outputs = []
            for o in outputs:
                # print("Original output is: ", o)
                # print("length of output is: ", len(o))
                # Strip leading/trailing whitespace from output
                cleaned_output = o.strip()

                # Remove the first 'prompt_length' characters from the output
                cleaned_output = cleaned_output[prompt_length-2:].strip()

                # print("length of cleaned output is: ", len(cleaned_output))
                # Append the cleaned output
                cleaned_outputs.append(cleaned_output)

            outputs = cleaned_outputs
            # outputs = [o.replace(prompt, '') for o in outputs]
        else:
            prompt = prompt.replace("<|im_start|>", "").replace("<|im_end|>",
                                                                "")  # For dbrx, because it won't output special token during generation
            prompt = prompt.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace("<|start_header_id|>",
                                                                                               "").replace(
                "<|end_header_id|>", "")  # For llama3
            outputs = [o.replace(prompt, '') for o in outputs]

        print("Prompt: ", prompt, "\n")
        print("Outputs: ", outputs, "\n")
        print("Ground truth: ", gt, "\n")
        # print the type of generated text
        # bool_sub, outputs_text_corrected = get_substring_after(outputs_text, pure_prompt)
        # if bool_sub:
        #     outputs_text = outputs_text_corrected
        #     print("Generated text (corrected): ", outputs_text, "\n")
        outputs_text = outputs[0]
        outputs = [extract_last_x_tokens(outputs_text, 100)]
        print("Generated text (corrected): ", outputs)
        inference_time_list.append(time_end - time_start)
        output_list.append(outputs)
        prompt_list.append(prompt)
        gt_list.append(gt)

    if memFree_Prompt_Config['eval_general'] and not args.eval_mode:
        mmlu_score = eval_mmlu(args,model=model, tokenizer=tokenizer, bloom_filter=bloom_filter)
        model_name_for_save = extract_model_name(model_name)
        save_path = f"SSU_Unlearn/res/mmlu_res/log_{model_name_for_save}_{intervention}_time_step_{time_step_num}.txt"
        save_filepath = os.path.join(args.base_dir, save_path)
        with open(save_filepath, "a") as f:
            print(
                f"{model_name}\t{intervention}\ttime_step_{time_step_num}\t{memFree_Prompt_Config['datatype']}\t{min_new_tokens}\tmmlu\t{mmlu_score:.4f}",
                file=f, flush=True)

    return output_list, prompt_list, gt_list, inference_time_list


def eval_infringement(model_name, data_type, prompt_list, gt_list, output_list, inference_time_list,
                       args):
    num_tests = memFree_Prompt_Config["num_tests"]
    agg_res = {}
    agg_res['model'] = model_name
    agg_res['num_tests'] = num_tests
    agg_res['context_len'] = min_new_tokens
    agg_res['completion_len'] = min_new_tokens

    rouge = evaluate.load('rouge')
    rouge_1, rouge_l, prompts = [], [], []

    intervention = memFree_Prompt_Config["intervention"]
    # eval semantic similarity
    semantic_sim = []
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=model_dir)

    best_rouge1_comps, best_rougeL_comps, best_verbatim_matching_comps, gts, matching_sequences = [], [], [], [], []
    best_rouge1_ids, best_rougeL_ids, best_verbatim_matching_ids = [], [], []
    best_verbatim_matching_ids, max_lengths, total_lengths = [], [], []
    # begin compute time
    # time_start = timeit.default_timer()
    for prompt, gt, outputs in zip(prompt_list, gt_list, output_list):
        best_verbatim_matching_id, matching_sequence, max_length, total_length = utils.find_common_sequences(outputs,
                                                                                                             gt)
        results = rouge.compute(predictions=outputs, references=[gt] * len(outputs), use_aggregator=False)

        # semantic simlarity
        ref_embeddings = model.encode([gt])
        pred_embeddings = model.encode(outputs)
        cos_sim = util.cos_sim(pred_embeddings, ref_embeddings).cpu().numpy().squeeze().tolist()
        if isinstance(cos_sim, float):
            cos_sim = [cos_sim]
        max_cos_sim = max(cos_sim)
        semantic_sim.append(max_cos_sim)
        # bp()

        max_rougeL = max(results['rougeL'])
        max_rouge1 = max(results['rouge1'])
        best_rougeL = outputs[results['rougeL'].index(max_rougeL)]
        best_rouge1 = outputs[results['rouge1'].index(max_rouge1)]
        best_verbatim_matching = outputs[best_verbatim_matching_id]
        prompts.append(prompt)
        rouge_1.append(max_rouge1)
        rouge_l.append(max_rougeL)
        best_rouge1_comps.append(best_rouge1)
        best_rougeL_comps.append(best_rougeL)
        best_verbatim_matching_comps.append(best_verbatim_matching)
        best_rouge1_ids.append(results['rouge1'].index(max_rouge1))
        best_rougeL_ids.append(results['rougeL'].index(max_rougeL))
        best_verbatim_matching_ids.append(best_verbatim_matching_id)
        max_lengths.append(max_length)
        total_lengths.append(total_length)
        gts.append(gt)
        matching_sequences.append(matching_sequence)

    data_type = f'{data_type}_low_ppl'
    df = pd.DataFrame({'prompt': prompts, 'gt': gts, 'rouge1': rouge_1, 'rougeL': rouge_l, 'semantic_sim': semantic_sim,
                       'best_rouge1': best_rouge1_comps, 'best_rougeL': best_rougeL_comps,
                       'best_verbatim_matching': best_verbatim_matching_comps,
                       'matching_sequence': matching_sequences,
                       'max_length': max_lengths, 'total_length': total_lengths,
                       'best_rouge1_ids': best_rouge1_ids, 'best_rougeL_ids': best_rougeL_ids,
                       "best_verbatim_matching_ids": best_verbatim_matching_ids, "inference_time": inference_time_list})

    train_or_test = memFree_Prompt_Config["train_or_test"]
    model_name = extract_model_name(model_name)
    if args.previous_time_steps:
        print("Saving results of bookings being unlearned during previous time steps")
        if 'mem_free' in intervention:
            path = f'SSU_Unlearn/res/output_previous_time_steps/{data_type}_single_{args.single_book}_comp_{model_name}_previous_time_step_{time_step_num}_intervention_{intervention}_{n}_no_context_{no_context}.csv'
        elif intervention == 'cad':
            path = f'SSU_Unlearn/res/output_previous_time_steps/{data_type}_single_{args.single_book}_comp_{model_name}_previous_time_step_{time_step_num}_intervention_{intervention}_{memFree_Prompt_Config["context_aware_decoding_alpha"]}_no_context_{no_context}.csv'
        else:
            path = f'SSU_Unlearn/res/output_previous_time_steps/{data_type}_single_{args.single_book}_comp_{model_name}_previous_time_step_{time_step_num}_intervention_{intervention}_no_context_{no_context}.csv'
    elif args.eval_mode:
        print("Evaluation mode is on")
        if 'mem_free' in intervention:
            path = f'SSU_Unlearn/res/output_norm_res/{data_type}_single_{args.single_book}_comp_{model_name}_time_step_{time_step_num}_intervention_{intervention}_{n}_no_context_{no_context}.csv'
        elif intervention == 'cad':
            path = f'SSU_Unlearn/res/output_norm_res/{data_type}_single_{args.single_book}_comp_{model_name}_time_step_{time_step_num}_intervention_{intervention}_{memFree_Prompt_Config["context_aware_decoding_alpha"]}_no_context_{no_context}.csv'
        else:
            path = f'SSU_Unlearn/res/output_norm_res/{data_type}_single_{args.single_book}_comp_{model_name}_time_step_{time_step_num}_intervention_{intervention}_no_context_{no_context}.csv'
    else:
        print("Evaluation mode is off")
        if 'mem_free' in intervention:
            path = f'SSU_Unlearn/res/output_res/{data_type}_single_{args.single_book}_comp_{model_name}_time_step_{time_step_num}_intervention_{intervention}_{n}_no_context_{no_context}_{train_or_test}.csv'
        elif intervention == 'cad':
            path = f'SSU_Unlearn/res/output_res/{data_type}_single_{args.single_book}_comp_{model_name}_time_step_{time_step_num}_intervention_{intervention}_{memFree_Prompt_Config["context_aware_decoding_alpha"]}_no_context_{no_context}_{train_or_test}.csv'
        else:
            path = f'SSU_Unlearn/res/output_res/{data_type}_single_{args.single_book}_comp_{model_name}_time_step_{time_step_num}_intervention_{intervention}_no_context_{no_context}_{train_or_test}.csv'

    path = os.path.join(args.base_dir, path)
    print("The path to save is: ", path)
    if memFree_Prompt_Config["no_overwrite"]:
        print("TEST1")
        counter = 1
        new_path = path
        while os.path.exists(new_path):
            base, extension = os.path.splitext(path)
            new_path = f"{base}_{counter}{extension}"
            counter += 1
        print("The new path is: ", new_path)
        df.to_csv(new_path)
    else:
        df.to_csv(path)

    res_process = add_metrics(df)

    agg_res['max_rouge1'] = df['rouge1'].max()
    agg_res['max_rougeL'] = df['rougeL'].max()
    agg_res['max_semantic_sim'] = df['semantic_sim'].max()
    agg_res['mean_rouge1'] = df['rouge1'].mean()
    agg_res['mean_rougeL'] = df['rougeL'].mean()
    agg_res['max_semantic'] = df['semantic_sim'].max()
    agg_res['mean_semantic'] = df['semantic_sim'].mean()
    agg_res['inference_time'] = sum(inference_time_list) / len(inference_time_list)
    agg_res['Minhash Similarity'] = res_process['Minhash Similarity'].mean()
    return agg_res

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def eval_mmlu(args, model, tokenizer, bloom_filter=None):
    print("Evaluating on MMLU...")

    intervention = memFree_Prompt_Config["intervention"]
    model_name = memFree_Prompt_Config["model_name"]
    prompt_instruction = ""
    subject2em = {}
    prior_processor = model._get_logits_processor
    mmlu_dir = os.path.join(args.base_dir, "SSU_Unlearn/CoTaEval/eval_data/mmlu")
    for subject in tqdm.tqdm(os.listdir(mmlu_dir)):
        all_em = []
        train_data = read_jsonl(os.path.join(args.base_dir, f"SSU_Unlearn/CoTaEval/eval_data/mmlu/{subject}/dev.jsonl"))
        # formulate the prompt
        prompt_orig = ""
        for ex in train_data:
            ex_instruction = """Question: {}\nChoices: A: {}, B: {}, C: {}, D: {},\nAnswer: {}\n\n"""
            ex_instruction = ex_instruction.format(ex['question'], ex['choices']['A'], ex['choices']['B'],
                                                   ex['choices']['C'], ex['choices']['D'], ex['answer'])
            prompt_orig += ex_instruction
        prompt_orig += prompt_instruction

        test_data = read_jsonl(os.path.join(args.base_dir,f"SSU_Unlearn/CoTaEval/eval_data/mmlu/{subject}/test.jsonl"))
        for ex in tqdm.tqdm(test_data[:50]):
            ex_test_instruction = """Question: {}\nChoices: A: {}, B: {}, C: {}, D: {},\n"""
            answer = ex['answer']
            test_data_prompt = ex_test_instruction.format(ex['question'], ex['choices']['A'], ex['choices']['B'],
                                                          ex['choices']['C'], ex['choices']['D'])
            prompt = prompt_orig + test_data_prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            context_len = inputs.input_ids.shape[1]

            if 'mem_free' in intervention:
                if any(element in model_name for element in
                       ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                    prompt = apply_prompt_template(prompt_template_style='sys_a', dataset=[prompt], eval_mode=True)[0]
                elif 'llama2-7b-hf' in model_name:
                    prompt = prompt
                elif 'llama3' in model_name or "Llama-3" in model_name:
                    prompt = apply_prompt_template(prompt_template_style='sys_a', dataset=[prompt], eval_mode=True,
                                                   model='llama3')[0]
                elif 'mistral' in model_name or "Mistral-7B" in model_name:
                    prompt = apply_prompt_template(prompt_template_style='sys_a', dataset=[prompt], eval_mode=True,
                                                   model='mistral')[0] # Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                bf_is_tokenized = "tokenized" in intervention
                match = re.search(r'\d+', bloom_filter)
                n = int(match[0])
                portrait = dataportraits.RedisBFSketch('localhost', 6379, bloom_filter, int(n))
                choice = intervention.split('-')[-1]
                if choice == 'consecutive':
                    if "tokenized" in intervention:
                        width = 2 * int(n) - 6
                    else:
                        width = 2 * int(n) - 1

                    def new_logits_processor(*args, **kwargs):
                        prior = prior_processor(*args, **kwargs)
                        if len(prior) == 1:
                            prior.pop()
                        prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait,
                                                                  bf_is_tokenized=bf_is_tokenized,
                                                                  tokenized_prompt=inputs, n=int(n), consecutive=True))
                        return prior
                else:
                    width = 3 * int(n)
                    acs_threshold = memFree_Prompt_Config["acs_threshold"]

                    def new_logits_processor(*args, **kwargs):
                        prior = prior_processor(*args, **kwargs)
                        if len(prior) == 1:
                            prior.pop()  # Remove the existing bloom_filter logits processor
                        prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait,
                                                                  bf_is_tokenized=bf_is_tokenized,
                                                                  tokenized_prompt=inputs, n=int(n), consecutive=False,
                                                                  acs_threshold=acs_threshold))
                        return prior
                model._get_logits_processor = new_logits_processor
                context_len = inputs.input_ids.shape[1]
                if context_len > 3500:
                    print("find examples with context length > 3500, continue")
                    continue
                generate_ids = model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False,
                                              num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,
                                              attention_mask=inputs.attention_mask)
            elif intervention == 'top_k':
                if any(element in model_name for element in
                       ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
                elif 'llama2-7b-hf' in model_name:
                    prompt = prompt
                elif 'llama3' in model_name or "Llama-3" in model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True,
                                                   model='llama3')[0]
                elif 'mistral' in model_name or "Mistral-7B" in model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True,
                                                   model='mistral')[0]
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                context_len = inputs.input_ids.shape[1]
                if context_len > 3500:
                    print("find examples with context length > 3500, continue")
                    continue
                generate_ids = model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False,
                                              num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,
                                              attention_mask=inputs.attention_mask)
            elif 'sys_prompt' in intervention:
                system_prompt_choice = intervention.split('-')[-1]
                if 'llama2' in model_name:
                    prompt = \
                    apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True)[
                        0]
                elif 'dbrx' in model_name:
                    prompt = \
                    apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True,
                                          model='dbrx')[0]
                elif 'llama3' in model_name or "Llama-3" in model_name:
                    prompt = \
                    apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True,
                                          model='llama3')[0]
                elif "mistral" in model_name or "Mistral-7B" in model_name:
                    prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True,
                                                   model='mistral')[0] # Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
                    # print("First, the prompt is: ", prompt)
                    # print("="*10)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                context_len = inputs.input_ids.shape[1]
                if context_len > 3500:
                    print("find examples with context length > 3500, continue")
                    continue
                generate_ids = model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False,
                                              num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,
                                              attention_mask=inputs.attention_mask)
            else:
                model.generation_config.context_aware_decoding_alpha = None
                if any(element in model_name for element in
                       ['llama2-7b-chat-hf', 'llama2-13b-chat-hf', 'llama2-70b-chat-hf']):
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
                elif 'dbrx' in model_name:
                    prompt = \
                    apply_prompt_template(prompt_template_style='dbrx', dataset=[prompt], eval_mode=True, model='dbrx')[
                        0]
                elif 'llama3' in model_name or "Llama-3" in model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True,
                                                   model='llama3')[0]
                elif "mistral" in model_name or "Mistral-7B" in model_name:
                    prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True,
                                                   model='mistral')[0]
                elif 'llama2-7b-hf' in model_name:
                    prompt = prompt
                else:
                    raise ValueError("Invalid model name")
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                context_len = inputs.input_ids.shape[1]
                if context_len > 3500:
                    print("find examples with context length > 3500, continue")
                    continue
                generate_ids = model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False,
                                              num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,
                                              attention_mask=inputs.attention_mask)

            outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if "mistral" in model_name or "Mistral-7B" in model_name:
                prompt = prompt.replace("[INST]", "").replace("[/INST]", "")
                prompt_length = len(prompt)
                # print("Secondly, Prompt is: ", prompt)
                # print("=" * 10)
                cleaned_outputs = []
                for o in outputs:
                    # Strip leading/trailing whitespace from output
                    cleaned_output = o.strip()
                    # print("Original output is: ", cleaned_output)
                    # Remove the first 'prompt_length' characters from the output
                    cleaned_output = cleaned_output[prompt_length-3:].strip()
                    # print("Cleaned output is: ", cleaned_output)
                    # Append the cleaned output
                    cleaned_outputs.append(cleaned_output)
                outputs = cleaned_outputs
                # outputs = [o.replace(prompt, '') for o in outputs]
            else:
                prompt = prompt.replace("<|im_start|>", "").replace("<|im_end|>",
                                                                    "")  # For dbrx, because it won't output special token during generation
                prompt = prompt.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace(
                    "<|start_header_id|>",
                    "").replace(
                    "<|end_header_id|>", "")
                outputs = [o.replace(prompt, '') for o in outputs]

            # prompt = prompt.replace("<|im_start|>", "").replace("<|im_end|>", "")
            # prompt = prompt.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace("<|start_header_id|>",
            #                                                                                    "").replace(
            #     "<|end_header_id|>", "")  # For llama3
            # # bp()
            # outputs = [o.replace(prompt, '') for o in outputs]
            outputs = outputs[0].split("\n")
            selected_outputs = [s for s in outputs if "Answer" in s]
            # print("Selected outputs: ", selected_outputs)
            if len(selected_outputs) == 0:
                all_em.append(0)
                continue
            else:
                outputs = selected_outputs[0]
            outputs = outputs.replace("Answer", "").strip(string.punctuation).strip()
            # if "mistral" in model_name or "Mistral-7B" in model_name:
            #     outputs = outputs[0].strip()
            print("Final Outputs: ", outputs)
            if (outputs not in ['A', 'B', 'C', 'D']):
                all_em.append(0)
                continue
            em = answer == outputs
            all_em.append(em)
        if len(all_em) == 0:
            continue
        else:
            em_subject = sum(all_em) / len(all_em)
            subject2em[subject] = em_subject
        print(subject2em)
    avg_em = sum(subject2em.values()) / len(subject2em)
    std_em = np.std(list(subject2em.values()))
    confidence_interval = 1.96 * std_em / np.sqrt(len(subject2em))
    print(f"Average EM: {avg_em}, std: {std_em}, confidence interval: {confidence_interval}")
    return avg_em

if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='Fine-tune LLM on specific books')
    parser.add_argument('--base_dir', type=str, help='The base dir')
    parser.add_argument('--model_dir', type=str, help='The directory where the model is saved')
    parser.add_argument('--time_step_num', type=int, help='The number of time steps for the unlearning process')
    parser.add_argument('--single_book', action='store_true',
                        help='Whether to use a single book to unlearn sequentially')
    parser.add_argument('--use_all', action='store_true',
                        help='Whether to use entire unlearning dataset for each time step')
    parser.add_argument('--eval_mode', action='store_true',
                        help="Whether to evaluate on the normal data (aka test data that are not being used in unlearning process)")
    parser.add_argument('--previous_time_steps', action='store_true',
                        help="Whether to evaluate data that has previously unlearned (set to true when time_step_num is greater than 1) only")
    parser.add_argument("--eval_mmlu_only", action='store_true',
                        help="Whether to evaluate only on MMLU (not applicable for memFree method)")
    args = parser.parse_args()

    n = memFree_Prompt_Config["n"]
    time_step_num = args.time_step_num

    train_or_test = memFree_Prompt_Config["train_or_test"]
    if args.eval_mmlu_only:
        print("Evaluating only on MMLU")
        mmlu_score = mmlu_only(n, base_model_name=memFree_Prompt_Config["model_name"] , model_dir=args.model_dir, args=args)
        print(f"MMLU score: {mmlu_score}")
    elif args.eval_mode:
        print("Running on eval mode; make sure eval_general is set to False")
        file_path = f'SSU_Unlearn/data_normal_csv/data_norm.csv'
    else:
        if args.single_book:
            if args.previous_time_steps:
                print("Evaluating on previously unlearned data")
                assert time_step_num > 1
                file_path = f'SSU_Unlearn/data_csv_single/time_step_{time_step_num}/time_step_{time_step_num}_combined_previous_tests.json'
            else:
                file_path = f'SSU_Unlearn/data_csv_single/time_step_{time_step_num}/time_step_{time_step_num}_{train_or_test}_dataset_unlearn.json'
        else:
            if args.previous_time_steps:
                print("Evaluating on previously unlearned data")
                assert time_step_num > 1
                file_path = f'SSU_Unlearn/data_csv/time_step_{time_step_num}/time_step_{time_step_num}_combined_previous_tests.json'
            else:
                file_path = f'SSU_Unlearn/data_csv/time_step_{time_step_num}/time_step_{time_step_num}_{train_or_test}_dataset_unlearn.json'

    if not args.eval_mmlu_only:
        print("File path being evaluated is: ", file_path)
        model_name = memFree_Prompt_Config["model_name"]
        is_instruct_model = memFree_Prompt_Config["is_instruct_model"]
        use_fine_tuned_model = False
        fine_tuned_model_name = None
        fine_tuned_filename = None
        if is_instruct_model:
            max_completion_tokens = 100
            min_new_tokens = 100
        else:
            max_completion_tokens = 200
            min_new_tokens = 200
        acs_threshold = memFree_Prompt_Config["acs_threshold"] # for non-consecutive case
        no_context = memFree_Prompt_Config["no_context"]

        model_dir = args.model_dir
        output_list, prompt_list, gt_list, inference_time_list =  main(n, file_path, model_name, use_fine_tuned_model, fine_tuned_model_name, fine_tuned_filename, model_dir, args)
        agg_res = eval_infringement(model_name, memFree_Prompt_Config["datatype"], prompt_list, gt_list, output_list,
                                    inference_time_list, args)
        print("File path being evaluated is: ", file_path)
        print(agg_res)
