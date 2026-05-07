"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import re
import dataportraits
from ipdb import set_trace as bp
import sys
from transformers import GenerationConfig
import shortuuid
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from lib.decoding_intervention import DataPortraitsLogitsProcessor, TopKPerturbationLogitsProcessor, DataPortraitsSkipLogitsProcessor
from lib.prompt_utils import apply_prompt_template

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
from config.llm_benchmark_config import MT_Bench_Config

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

def run_eval(
    model_path,
    model_name,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_name,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_name,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
):
    # model, tokenizer = load_model(
    #     model_path,
    #     model_name=model_name,
    #     revision=revision,
    #     device="cuda",
    #     num_gpus=num_gpus_per_model,
    #     max_gpu_memory=max_gpu_memory,
    #     dtype=dtype,
    #     load_8bit=False,
    #     cpu_offloading=False,
    #     debug=False,
    # )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    access_token = os.environ.get('HF_ACCESS_TOKEN')
    if model_name in ["meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
        print(f"Loading model {model_name} with model_path {model_path}")

        if intervention_choice[0] == 'unlearning':
            if MT_Bench_Config["use_quantization"]:
                model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config,
                                                             device_map="auto", token=access_token,
                                                             cache_dir=args.model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path,
                                                             device_map="auto",
                                                             token=access_token,
                                                             torch_dtype=torch.float16,
                                                             cache_dir=args.model_path)
        else:
            modified_base_model_name = f"Mistral-7B-Instruct-v0.3_time_step_10_intervention_unlearning_gd_none"
            model_checkpoint_path = os.path.join(args.model_path, modified_base_model_name)
            print("Loading Model checkpoint path: ", model_checkpoint_path)
            if MT_Bench_Config["use_quantization"]:
                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, quantization_config=bnb_config,
                                                             device_map="auto", token=access_token,
                                                             cache_dir=args.model_path)
            else:

                model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path,
                                                             device_map="auto",
                                                             token=access_token,
                                                             torch_dtype=torch.float16,
                                                             cache_dir=args.model_path)
                # model = AutoModelForCausalLM.from_pretrained(model_name,
                #                                              device_map="auto",
                #                                              token=access_token,
                #                                              torch_dtype=torch.float16,
                #                                              cache_dir=args.model_path)


        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    if "mistral" in model_name or "Mistral-7B" in model_name:
        tokenizer.pad_token = "<pad>"
    else:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    if 'dbrx' in model_id:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    prior_processor = model._get_logits_processor
    if intervention != 'cad':
        model.generation_config.context_aware_decoding_alpha = None # Here we add this to avoid error for the non-cad situation.
    if intervention == 'mem_free_new':
        print("intervention is mem_free_new")
        model.generation_config.mem_free_new = True
    else:
        model.generation_config.mem_free_new = False

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        print("Temperature is", temperature)
        choices = []
        
        for i in range(num_choices):
            torch.manual_seed(i)
            if 'llama2' in model_id:
                model_id_conv = "llama-2-7b-hf"
            elif 'dbrx' in model_id:
                model_id_conv = 'dbrx'
            elif 'llama3' in model_id or "Llama-3" in model_id:
                model_id_conv = 'llama-3'
            elif 'mistral' in model_id:
                model_id_conv = 'mistral'
            else:
                model_id_conv = model_id
            print("Model id conv is", model_id_conv)
            conv = get_conversation_template(model_id_conv)
            turns = []
            for j in range(len(question["turns"])):
                n = MT_Bench_Config["n"]
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # if "llama2-7b-hf" in model_id:
                #     prompt = prompt.replace("[INST]", "").replace("[/INST]", "").strip()
                # bp()
                # swj: add intervention here:
                print("The prompt is", prompt)
                if 'mem_free' in intervention:
                    if "tokenized" in intervention:
                        bloom_filter = f'gutenberg_books_time_step_{time_step_num}_tokenized.{6 * n}-{6 * n}.bf'
                    else:
                        bloom_filter = f'gutenberg_books_time_step_{time_step_num}.{n}-{n}.bf'
                    bf_is_tokenized = "tokenized" in intervention
                    choice = intervention.split('-')[-1]
                    if "llama2-7b-hf" in model_id:
                        prompt = prompt
                    else:
                        prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
                    matches = re.findall(r'\d+', bloom_filter)
                    if len(matches) > 1:
                        n = matches[1]
                    else:
                        n = matches[0]
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    portrait = dataportraits.RedisBFSketch('localhost', 6379, bloom_filter, int(n))
                    if choice == 'consecutive':
                        if "tokenized" in intervention:
                            # width = 2 * int(n) - 5
                            width = 2 * int(n) - 6
                        else:
                            width = 2 * int(n) - 1
                        def new_logits_processor(*args, **kwargs):
                            prior = prior_processor(*args, **kwargs)
                            prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized, tokenized_prompt=inputs, n=int(n), consecutive=True))
                            return prior
                    else:
                        width = 3 * int(n)
                        acs_threshold = MT_Bench_Config["acs_threshold"]
                        def new_logits_processor(*args, **kwargs):
                            prior = prior_processor(*args, **kwargs)
                            prior.append(DataPortraitsLogitsProcessor(prompt, width, tokenizer, portrait, bf_is_tokenized=bf_is_tokenized, tokenized_prompt=inputs, n=int(n), consecutive=False, acs_threshold=acs_threshold))
                            return prior
                    model._get_logits_processor = new_logits_processor
                    # Generate text completions

                elif intervention == 'top_k':
                    if 'llama2-7b-hf' in args.model_id:
                        prompt = prompt
                    else:
                        prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    std = 3
                    def new_logits_processor(*args, **kwargs):
                        prior = prior_processor(*args, **kwargs)
                        prior.append(TopKPerturbationLogitsProcessor(tokenizer, model, std))
                        return prior
                    model._get_logits_processor = new_logits_processor
                    
                elif 'sys_prompt' in intervention:
                    system_prompt_choice = intervention.split('-')[-1]
                    if 'llama2' in model_id:
                        prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True)[0]
                    elif 'dbrx' in model_id:
                        prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], eval_mode=True, model='dbrx')[0]
                    elif 'llama3' in model_id:
                        prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt],
                                                       eval_mode=True, model='llama3')[0]
                    elif 'mistral' in model_id or 'mistral7b' in model_id:
                        prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt],
                                                       eval_mode=True, model='mistral')[0] # Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
                else: # includes vanilla case as well as FT base model case.
                    if "llama2-7b-hf" in model_id:
                        prompt = prompt
                    elif "llama2" in model_id:
                        prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True)[0]
                    elif "dbrx" in model_id:
                        prompt = apply_prompt_template(prompt_template_style='dbrx', dataset=[prompt], eval_mode=True, model="dbrx")[0]
                    elif 'llama3' in model_id:
                        prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model="llama3")[0]
                    elif 'mistral' in model_id or 'mistral7b' in model_id:
                        prompt = apply_prompt_template(prompt_template_style='none', dataset=[prompt], eval_mode=True, model="mistral")[0]
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                input_ids = tokenizer([prompt]).input_ids
                # if temperature < 1e-4:
                #     do_sample = False
                # else:
                #     do_sample = True
                do_sample = False
                # some models may error out when generating long outputs
                # try:
                keywords = ["chat", "Chat", "instruct", "Instruct"]
                if any(keyword in model_name for keyword in keywords):
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                        pad_token_id=tokenizer.pad_token_id
                    )
                else:
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        min_new_tokens=len(input_ids[0])+1,
                        max_new_tokens=max_new_token,
                        pad_token_id=tokenizer.pad_token_id
                    )
                # print("output_ids is", output_ids)
                # print("length of output_ids[0]", len(output_ids[0]))
                # print("output_ids is", output_ids)
                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    # print("input_ids[0] is", input_ids[0])
                    # print("len(input_ids[0]) is", len(input_ids[0]))
                    output_ids = output_ids[0][len(input_ids[0]) :]

                # print("Output_ids is", output_ids)
                # print("The length of output_ids is", len(output_ids))
                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                print("Output is", output)
                if conv.stop_str and isinstance(conv.stop_str, list):
                    stop_str_indices = sorted(
                        [
                            output.find(stop_str)
                            for stop_str in conv.stop_str
                            if output.find(stop_str) > 0
                        ]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
                # except RuntimeError as e:
                #     print("ERROR question ID: ", question["question_id"])
                #     output = "ERROR"
                print("The output (modified) is", output)
                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        print("Created a directory for the answer file")
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="The base directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    # parser.add_argument(
    #     "--model-id", type=str, required=True, help="A custom name for the model."
    # )
    parser.add_argument('--time_step_num', type=int, help='The number of time steps for the unlearning process')

    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    # parser.add_argument("--intervention", type=str, default="none", choices=["none", "none_greedy",
    #                                                                         "top_k",
    #                                                                         "mem_free-consecutive", "mem_free-non_consecutive",
    #                                                                         "mem_free_tokenized-consecutive",
    #                                                                         "mem_free_new",
    #                                                                         "sys_prompt-sys_a", "sys_prompt-sys_b", "sys_prompt-sys_c",
    #                                                                         "sys_prompt-bing", "sys_prompt-copilot",  "sys_prompt-dbrx",
    #                                                                         "sys_prompt-sys_none",
    #                                                                         "uncopy", "uncopy_merge",
    #                                                                         "uncopy_fact", "uncopy_summary",
    #                                                                         "cad",
    #                                                                         "unlearning-gradient_ascent", "unlearning-dpo"])
    # parser.add_argument("--bf_type", type=str, default="newsqa", choices=["newsqa", "booksum"])

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    intervention = MT_Bench_Config["intervention"]
    bf_type = MT_Bench_Config["bf_type"]
    time_step_num = args.time_step_num
    model_name = MT_Bench_Config["model_name"]
    model_id = MT_Bench_Config["model_id"]
    model_id = f"{model_id}_time_step_{time_step_num}_{intervention}"

    intervention_choice = intervention.split('_')
    if intervention_choice[0] == "unlearning":
        modified_base_model_name = extract_model_name(model_name)
        modified_base_model_name = f"{modified_base_model_name}_time_step_{time_step_num}_intervention_{intervention}"
        model_checkpoint_path = os.path.join(args.model_path, modified_base_model_name)
        model_path = model_checkpoint_path
    else:

        # model_path = os.path.join(args.model_path, model_name)
        model_path = model_name
    print("Model path: ", model_path, "with model id: ", model_id)
    question_file = f"SSU_Unlearn/CoTaEval/eval/FastChat_new/fastchat/llm_judge/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"SSU_Unlearn/CoTaEval/eval/FastChat_new/fastchat/llm_judge/data/{args.bench_name}/model_answer/{model_id}.jsonl"

    question_file = os.path.join(args.base_dir, question_file)
    answer_file = os.path.join(args.base_dir, answer_file)
    print(f"First, Output to {answer_file}")

    run_eval(
        model_path=model_path,
        model_name=model_name,
        model_id=MT_Bench_Config["model_id"],
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
    )

    reorg_answer_file(answer_file)
    print(f"Last, Output to {answer_file}")
