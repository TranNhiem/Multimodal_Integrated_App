'''
@TranNhiem 2023/06/09
For Finetuning Larger LLMs Model 


# 1. First Step: Download Directly from 
    Quantization GPTQ --> Quantized LLM Models to 4 Bits 
    https://github.com/IST-DASLab/gptq
    https://huggingface.co/blog/chatbot-amd-gpu 

# 2. Second Step: Using LoRA to FineTune LLM via Low Bit Percision 
    # https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing#scrollTo=s6f4z8EYmcJ6
    + https://huggingface.co/blog/4bit-transformers-bitsandbytes 

# 3. Further Optimization FineTuning via Deepspeed & Triton (Gradient Checkpointing) & Sparse LLMs
    + DeepSpeed Implementation

'''

import torch
import os 
from os.path import exists, join, isdir
import transformers 
import bitsandbytes as bnb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer




max_memory_MB = 80000
model_name_or_path="huggyllama/llama-7b" ## Setting Model later 
fp16 = False
bf16=True 
cache_dir="/data/rick/pretrained_weights/LLaMA/"
bits=4
double_quant=True
lora_modules= "all" 
lora_r=64
lora_alpha=16
lora_dropout=0.0
quant_type ="nf4" 
trust_remote_code=False #"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
use_auth_token=False # "help": "Enables using Huggingface auth token from Git Credentials."
full_finetune=False # "help": "Finetune the entire model without adapters."
checkpoint_dir=None 
gradient_checkpointing=True #"help": 'Use gradient checkpointing. You want to use this.'

## Function to load the model
def get_model(model_name_or_path, cache_dir, lora_alpha, lora_dropout, lora_r):
    
    n_gpus = torch.cuda.device_count()
    max_memory = f'{max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"
    
    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}
    if full_finetune: assert bits in [16, 32]
    print(f'loading base model {model_name_or_path}...')
    compute_dtype = (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=bits ,
            #load_in_8bit=bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type=quant_type,
        ),
        torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)),
        trust_remote_code=trust_remote_code,
        use_auth_token=use_auth_token
    )

    if compute_dtype == torch.float16 and bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32))

    if not full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            print(f'adding LoRA modules...')
            modules = find_all_linear_names( model)
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    return model
    
## Helper Function to check the number of trainable parameters


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

#model=get_model(model_name_or_path, bits=bits, fp16=fp16, bf16=bf16, double_quant=double_quant, quant_type=quant_type, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, gradient_checkpointing=gradient_checkpointing, full_finetune=full_finetune, checkpoint_dir=checkpoint_dir, trust_remote_code=trust_remote_code, use_auth_token=use_auth_token)
model=get_model(model_name_or_path, cache_dir, lora_alpha, lora_dropout, lora_r)

print_trainable_parameters(model)

modules=find_all_linear_names(model)
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    #target_modules=["g_proj", "v_proj",]
    target_modules=modules,
)
model = get_peft_model(model, peft_config)


for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


##We will also pre-process the model by upcasting the layer norms in float 32 for more stable trainin
# for name, module in model.named_modules():
#     if "norm" in name:
#         module = module.to(torch.float32)


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    cache_dir=cache_dir,
    padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    tokenizer_type='llama' if 'llama' in model_name_or_path else None, # Needed for HF name change
    use_auth_token=use_auth_token,
)
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
DEFAULT_PAD_TOKEN = "[PAD]"
if tokenizer._pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )


from datasets import load_dataset

dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name, split="train")

from transformers import TrainingArguments

output_dir = "./results"
per_device_train_batch_size = 16
gradient_accumulation_steps = 4
optim = "adamw_bnb_8bit"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "constant"

# use_wandb=True
wandb_project = "Vietnamese_LLMs"
wandb_run_name = "SFT_LLaMA_65B_QLORA_Alpaca_Vi"
wandb_watch = "all"  # options: false | gradients | all
wandb_log_model= "true"  # options: false | true
   # Check if parameter passed or if set within environ

use_wandb = len(wandb_project) > 0 or (
    "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
)
# Only overwrite environ if wandb param passed
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
if len(wandb_watch) > 0:
    os.environ["WANDB_WATCH"] = wandb_watch
if len(wandb_log_model) > 0:
    os.environ["WANDB_LOG_MODEL"] = wandb_log_model

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb" if use_wandb else None,
    run_name=wandb_run_name if use_wandb else None,
)

from trl import SFTTrainer

max_seq_length = 2048

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,

)


trainer.train()