<p align="center">

<img src="assets/ssu.png" alt="SSU_Unlearn"/>

</p>

<div align="center">

# <img src="assets/ssu_avatar.png" style="height: 1em; vertical-align: middle;" /> SSU: A Framework for Sequential Unlearning of Copyrighted Contents from LLMs.

</div>

<div align="center">

</div>

## Overview
 Pre-trained Large Language Models (LLMs) have demonstrated remarkable capabilities but also pose risks by learning and generating copyrighted material, leading to significant legal and ethical concerns. In real-world scenarios, model owners need to continuously address copyright infringement as new requests for content removal emerge at different time points. This leads to the need for sequential unlearning, where copyrighted content is removed sequentially as new requests arise. Despite its practical relevance, sequential unlearning in the context of copyright infringement has not been rigorously explored in existing literature. To address this gap, we propose Stable Sequential Unlearning (SSU), a novel framework designed to unlearn copyrighted content from LLMs over multiple time steps. Our approach works by identifying and removing specific weight updates in the model's parameters that correspond to copyrighted content. We improve unlearning efficacy by introducing random labeling loss and ensuring the model retains its general-purpose knowledge by adjusting targeted parameters. Experimental results show that SSU achieves an effective trade-off between unlearning efficacy and general-purpose language abilities, outperforming existing baselines.

You can check out the paper [here](https://arxiv.org/abs/2406.10952), which has been accepted to NAACL-Findings 2025.
## ⚙️ Installation
You need to install packages described in [requirements.txt](requirements.txt). You will also need
a .env file to store you HF_ACCESS_TOKEN to download Llama models.

## 🛠️ Dataset Setup

To begin, obtain the `.txt` versions of the books. You can either purchase them or download them from public sources, such as [Project Gutenberg](https://gutenberg.org/), or you can crawl and preprocess
these books by following [gutenberg](https://github.com/pgcorpus/gutenberg).

### Directory Structure
1. **Create a main directory called {PATH_TO_DATA}**: This will store the `.txt` files of the books.
2. **Organize files by time steps**: Inside the main directory, create subdirectories for each time step, named as `time_step_{x}` (e.g., `time_step_1`, `time_step_2`, etc.). Place the respective `.txt` files in these subdirectories.

### Commands to Run

Once the directory structure is set up, execute the following commands in sequence:

```bash
python create_training_data_csv.py --input_dir {PATH_TO_DATA}
python create_training_data_bf.py
python generate_json_train_test.py
python generate_json.py
python combine_previous_json.py
```

## 🏃Fine-tuning 
To fine-tune the model, you can use the [fine_tune_books.py](fine_tune_books.py) script. 

The script create_training_data.py requires the following parameters:

* --model_dir: Specifies the directory where your models are saved locally. This is essential for loading pre-trained or fine-tuned models.
* --data_dir: Specifies the directory containing all the books you have saved. This directory will be used to load the data for the unlearning process.
* --book_corpus_norm_data: Specifies the directory of the book corpus that does not contain the books you want to unlearn. This is used for GA-based methods.
* --Lora: A flag to use LoRA (Low-Rank Adaptation) parameterization. If set, the script will use LoRA for fine-tuning. Default is False.
* --time_step_num: An optional parameter to specify the time step number. If not specified, the script will fine-tune all the books in the data directory.


You also need to adjust the fine-tuning config file [fine_tune_config.py](config/fine_tune_config.py):
* base_model_name: Specifies the base model name. This is the base model you want to unlearn.
* use_quantization: True if you want to use quantization, False otherwise (Please set to False because TV subtraction will be inaccurate using quantization).
* time_step_num: The time step number you want to fine-tune.
* random_loss_epsilon: Specifies the epsilon value for the random labeling loss.
* num_std_dev : 0 if you just want to use mean, or however many standard deviations away from mean you want to use for the saliency-based weight update.
* batch_size: The batch size for fine-tuning.
* lr: The learning rate for fine-tuning.
* max_unlearn_steps: The number of epochs for each unlearning steps.
* gradient_accumulation_steps: The number of gradient accumulation steps.
* ga_factual_epsilon: The epsilon value for the factual loss term for GA Difference.
* npo_beta: The beta value for the NPO.
* random_loss_pairs: Number of mismatch pairs for SSU.
* intervention: The intervention method for fine-tuning. 

Available options are "unlearning_ga_none" (GA), "unlearning_npo_none" (NPO), "unlearning_ga_mismatch" (Gradient Difference), "unlearning_tv_none" (Pure TV),
"unlearning_tv_ssu" (SSU). Note that when running NPO, you should first obtain a oracle model. In this case, you should use 
'unlearning_npo_ref' as the intervention method. 

In addition, if you are interested in running ablation studies, you can use the following interventions:
"unlearning_tv_ssu_no_weight_saliency" and "unlearning_tv_ssu_no_random_loss".


Lastly, in order to fine-tune a model on $D_f$, you should use the intervention "unlearning_gd_none". 


## 📊 Evaluation

Please install [CoTaEval](https://github.com/boyiwei/CoTaEval/tree/main) to download MMLU dataset,
setup Bloom filters for MemFree Decode, and setup MT-Bench running environment. For MT-Bench, you will need to 
slightly modify [gen_model_answer.py](https://github.com/boyiwei/CoTaEval/blob/main/eval/FastChat_new/fastchat/llm_judge/gen_model_answer.py) to 
use the unlearned model. 

After setting up the evaluation environment, you can run the [evaluate_unlearn.py](evaluate_unlearn.py) script to evaluate the unlearning performance.

The script [evaluate_unlearn.py](evaluate_unlearn.py) requires the following parameters:
* --base_dir: The directory to this package.
* --model_dir: Specifies the directory where your models are saved locally. This is essential for loading pre-trained or fine-tuned models.
* --time_step_num: The time step number you want to evaluate.
* --single_book: A flag to evaluate a single book at each time step (If set to False, please update file_path variable in the code).
* --use_all: A flag to use entire unlearning dataset for each time step (user can choose to unlearn a part of it (by splitting into 'training' and 'testing' set to save computational resources). 
* --eval_mode: True if we only evaluate $D_{nor}$
* --eval_mmlu_only: True if we only evaluate MMLU.

You also need to adjust the evaluation config file [metrics_config.py](config/metrics_config.py):
* model_name: Specifies the base model name. This is the base model you want to unlearn.
* is_instruct_model: True if the model_name is a instruct model, False otherwise.
* use_quantization: True if you want to use quantization, False otherwise (Please set it to False).
* train_or_test: Select training or testing set (Ignore it if you have custom file_path).
* model_dir: Specifies the directory where your models are saved locally. This is essential for loading pre-trained or fine-tuned models.
* datatype: Specifies the datatype for evaluation. Default is 'gutenberg_books'.
* num_test: If use_all is set to False, you need to specify number of testing data.
* eval_general: True if you want to evaluate MMLU after running performance on unlearned books. Default is False.
* n : Choice of n-grams for MemFree Decoding.
* no_context: True if you want to provide no context for MemFree Decoding in the system prompt.
* no_overwrite: True if you do want to overwrite the existing results.
* acs_threshold: The threshold for ACS.
* intervention: The intervention method for evaluation.

Specifically, available intervention methods are: "sys_prompt-sys_none", "sys_prompt-sys_a", "sys_prompt-dbrx",
"unlearning_ga_none", "unlearning_npo_none", "unlearning_ga_idk", "unlearning_ga_mismatch",
"unlearning_tv_none", "unlearning_tv_ssu", "mem_free_tokenized_consecutive".

If you want to run ablation studies, you can set intervention to be "unlearning_tv_ssu_no_weight_saliency",
  or  "unlearning_tv_ssu_no_random_loss".


## 🙏 Acknowledgements and Citation 🙏

We thank the following open-source repositories. If aspects of these repositories appearing in SSU are useful to you in your research, we ask that you consider citing the accompanying papers.
```angular2html
[1] https://github.com/lm-sys/FastChat
[2] https://github.com/pgcorpus/gutenberg
[3] https://github.com/boyiwei/CoTaEval
[4] https://github.com/huggingface/transformers
```
  
If you find SSU useful in your research, please consider citing our [paper](https://arxiv.org/abs/2406.10952):
```
@article{dou2024avoiding,
  title={Avoiding Copyright Infringement via Machine Unlearning},
  author={Dou, Guangyao and Liu, Zheyuan and Lyu, Qing and Ding, Kaize and Wong, Eric},
  journal={arXiv preprint arXiv:2406.10952},
  year={2024}
}
```
