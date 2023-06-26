import time
from functools import lru_cache
import os
import torch
import gradio as gr
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
## Loading The FineTuned LoRa Adapter Model 
from peft import PeftModel, PeftConfig
import bitsandbytes as bnb
from transformers import  AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

base_model_3="bigscience/bloomz-1b7"

checkpoint_path_3="/data/rick/pretrained_weights/BLOOMZ/Alpaca_CN_500K/1b7_Bloomz_based/checkpoint-48400/"
cache_dir_="/data/rick/pretrained_weights/BLOOMZ/"


# model_3 = AutoModelForCausalLM.from_pretrained(
#     base_model_3,
#     cache_dir=cache_dir_,
#     #load_in_8bit=True, ## Currently RTX 1080Ti not working 
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# tokenizer_3 = AutoTokenizer.from_pretrained(base_model_3,  torch_dtype=torch.float16,cache_dir=cache_dir_,)#

# model_3 = prepare_model_for_int8_training(model_3)
# # bits=16
# # modules = find_all_linear_names(bits, model)
    
# config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules= ["query_key_value"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# model_3 = get_peft_model(model_3, config)

# checkpoint_name_3 = os.path.join(checkpoint_path_3, "pytorch_model.bin")
# print(f"Restarting from {checkpoint_name_3}")
# adapters_weights_3 = torch.load(checkpoint_name_3)
# for name, param in model_3.named_parameters():
#     #if name == "base_model.model.transformer.h.1.self_attention.query_key_value.lora_A.default.weight":
#     weight_tensor = adapters_weights_3[name]  # Get the corresponding tensor from weight_value
#     param.data = weight_tensor  # Replace the parameter tensor with weight_tensor


# #del (adapters_weights)
# model_3.to('cuda')

base_model_1="huggyllama/llama-13b"

checkpoint_path_1="/data/rick/pretrained_weights/LLaMA/alpaca_gpt4_llama_13B/checkpoint-2800"

cache_dir_llama="/data/rick/pretrained_weights/LLaMA/"
model_1 = AutoModelForCausalLM.from_pretrained(
    base_model_1,
    cache_dir=cache_dir_llama,
    #load_in_8bit=True, ## Currently RTX 1080Ti not working 
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer_1 = AutoTokenizer.from_pretrained(base_model_1,  torch_dtype=torch.float16,cache_dir=cache_dir_,)#

model_1 = prepare_model_for_int8_training(model_1)
# bits=16
# modules = find_all_linear_names(bits, model)
    
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules= ["g_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model_1 = get_peft_model(model_1, config)

checkpoint_name_1 = os.path.join(checkpoint_path_1, "pytorch_model.bin")
print(f"Restarting from {checkpoint_name_1}")
adapters_weights_1 = torch.load(checkpoint_name_1)
for name, param in model_1.named_parameters():
    #if name == "base_model.model.transformer.h.1.self_attention.query_key_value.lora_A.default.weight":
    weight_tensor = adapters_weights_1[name]  # Get the corresponding tensor from weight_value
    param.data = weight_tensor  # Replace the parameter tensor with weight_tensor


#del (adapters_weights)
model_1.to('cuda')



text_generation_pipe= pipeline("text-generation", model=model_1, tokenizer=tokenizer_1, num_workers=10)


def get_model_tokenizer(model_id):
    if model_name=="Alpha-1B1":
        base_model="ckip-joint/bloom-1b1-zh"
        if checkpoint_path is None:
            checkpoint_path="/data/rick/pretrained_weights/BLOOMZ/Alpaca_CN_500K/1b1_Bloomz_cn_based/checkpoint-23000/" # Path to save model weight to Disk

    elif model_name=="Alpha-7B1":
        base_model="bigscience/bloomz-7b1"
        if checkpoint_path is None:
            checkpoint_path="/data/rick/pretrained_weights/BLOOMZ/Alpaca_CN_500K/7b1_Bloomz_based/checkpoint-44000/"
    elif model_name=="Alpha-1B7":
        base_model="bigscience/bloomz-1b7"
        if checkpoint_path is None:
            checkpoint_path="/data/rick/pretrained_weights/BLOOMZ/Alpaca_CN_500K/1b7_Bloomz_based/checkpoint-48400/"
    else:
        raise ValueError(f"This Model {model_name} is not Supported")
    cache_dir_="/data/rick/pretrained_weights/BLOOMZ/"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache_dir_,
        #load_in_8bit=True, ## Currently RTX 1080Ti not working 
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model,  torch_dtype=torch.float16,cache_dir=cache_dir_,)#

    model = prepare_model_for_int8_training(model)
    # bits=16
    # modules = find_all_linear_names(bits, model)
        
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules= ["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    
    checkpoint_name = os.path.join(checkpoint_path, "pytorch_model.bin")
    print(f"Restarting from {checkpoint_name}")
    adapters_weights = torch.load(checkpoint_name)
    for name, param in model.named_parameters():
        #if name == "base_model.model.transformer.h.1.self_attention.query_key_value.lora_A.default.weight":
        weight_tensor = adapters_weights[name]  # Get the corresponding tensor from weight_value
        param.data = weight_tensor  # Replace the parameter tensor with weight_tensor
    
   
    #del (adapters_weights)
    model.to('cuda')
    return model, tokenizer


def run_generation(text, model_id, 
                        max_new_tokens=100, 
                        temperature=0.1, 
                        do_sample=False, 
                        alpha=0.0,
                        top_k=0,
                        num_beams=1,
                        top_p=0.0,
                        seed=0 ):

    #text2text_generator, tokenizer=get_model_tokenizer(model_id)
    text2text_generator=text_generation_pipe
    tokenizer=tokenizer_1
    text = text.strip()
    start = time.time_ns()
    text=f"### prompt: {text}. \n### response: "
    out_text = text2text_generator(
                            text, max_length=max_new_tokens, 
                            temperature=temperature, 
                            do_sample=do_sample,
                            eos_token_id = tokenizer.eos_token_id,
                            bos_token_id = tokenizer.bos_token_id,
                            pad_token_id = tokenizer.pad_token_id,
                            ## Fixed Arugment
                            early_stopping=True,
                            num_return_sequences=1,
                            num_beams=num_beams,
                            penalty_alpha=alpha or None,
                            top_k=top_k or None,
                            top_p=top_p or None,
                         )[0]['generated_text']
    print(out_text)
    end = time.time_ns()
    contrastive_time = (end - start) / 1e6
    #out_text = "<p>" + out_text + "</p>"
    #out_text = out_text.replace(text, text + "<b><span style='background-color: #ffffcc;'>")
    #out_text = out_text +  "</span></b>"
    #out_text = out_text.replace("\n", "<br>")

    return out_text, contrastive_time


def generate_beam_search(text, model_id, max_new_tokens, alpha, k, num_beams):
    contrastive_text, contrastive_time = run_generation(text, model_id, max_new_tokens, alpha=alpha, top_k=k)
    beam_search_text, beam_search_time = run_generation(text, model_id, max_new_tokens, num_beams=num_beams)
    return contrastive_text, contrastive_time, beam_search_text, beam_search_time


def generate_top_k(text, model_id, max_new_tokens, alpha, k, top_k,temperature, seed):
    contrastive_text, contrastive_time = run_generation(text, model_id, max_new_tokens, alpha=alpha, top_k=k)
    top_k_text, top_k_time = run_generation(
        text, model_id, max_new_tokens, top_k=top_k, seed=seed, do_sample=True, temperature=temperature
    )
    return contrastive_text, contrastive_time, top_k_text, top_k_time


def generate_top_p(text, model_id, max_new_tokens, alpha, k, top_p, seed):
    contrastive_text, contrastive_time = run_generation(text, model_id, max_new_tokens, alpha=alpha, top_k=k)
    top_p_text, top_p_time = run_generation(
        text, model_id, max_new_tokens, top_p=top_p,temperature=0.2, seed=seed, do_sample=True
    )
    return contrastive_text, contrastive_time, top_p_text, top_p_time

##Helper Read the HTML title 
def read_content(file_path) :
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

demo = gr.Blocks()
with demo:
    gr.HTML(read_content("/data/rick/LLM/Multimodal_Integrated_App/Language/Decoding_Strategies.html"))
    with gr.Tabs():
        with gr.TabItem("vs. Beam Search"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Inputs ✍️")
                    gr.Markdown("General options:")
                    #model_id = gr.Text(value="facebook/galactica-6.7b", label="Model Repository")
                    model_id = gr.Dropdown(choices= ["Alpha-7B1", "Alpha-1B7","Alpha-1B1" ], value="Alpha-1B7", label="Choosing LLM", show_label=True)

                    input_text = gr.Textbox(value="你能告訴我一步一步成為一個更好的計算機科學工程。", lines=5, label="Input Text")
                    max_new_tokens = gr.Slider(value=1024, minimum=100, maximum=2048, label="New tokens to generate")
                    gr.Markdown("Contrastive Search options:")
                    alpha = gr.Slider(value=0.6, minimum=0.01, maximum=1.0, step=0.01, label="Alpha")
                    k = gr.Slider(value=6, minimum=1, maximum=20, step=1, label="K")
                    gr.Markdown("Beam Search options:")
                    num_beams = gr.Slider(value=4, minimum=1, maximum=16, step=1, label="Number of beams")
                    generate_button = gr.Button(value="Generate", label="Generate")

                with gr.Column():
                    gr.Markdown("## Outputs 🤖")
                    gr.Markdown("Contrastive Search generation:")
                    text_contrastive = gr.Textbox(value="", label="")
                    time_contrastive = gr.Number(value=0.0, precision=1, label="Generation time (ms)")
                    gr.Markdown("Beam Search generation:")
                    text_beam_search = gr.Textbox(value="", label="")
                    time_beam_search = gr.Number(value=0.0, precision=1, label="Generation time (ms)")

            # actions
            generate_button.click(
                fn=generate_beam_search,
                inputs=[input_text, model_id, max_new_tokens, alpha, k, num_beams],
                outputs=[text_contrastive, time_contrastive, text_beam_search, time_beam_search]
            )

        with gr.TabItem("vs. Top K Sampling"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Inputs ✍️")
                    gr.Markdown("General options:")
                    #model_id = gr.Text(value="facebook/galactica-6.7b", label="Model Repository")
                    model_id = gr.Dropdown(choices= ["Alpha-7B1", "Alpha-1B7","Alpha-1B1" ], value="Alpha-1B7", label="Choosing LLM", show_label=True)

                    input_text = gr.Textbox(value="你能告訴我一步一步成為一個更好的計算機科學工程。", lines=5, label="Input Text")
                    max_new_tokens = gr.Slider(value=1024, minimum=1, maximum=2048, label="New tokens to generate")
                    gr.Markdown("Contrastive Search options:")
                    alpha = gr.Slider(value=0.6, minimum=0.01, maximum=1.0, step=0.01, label="Alpha")
                    k = gr.Slider(value=6, minimum=1, maximum=20, step=1, label="K")
                    gr.Markdown("Sampling options:")
                    temperature = gr.Slider(value=0.7, minimum=0.1, maximum=1, step=1, label="K")
                    gr.Markdown("Sampling options:")
                    top_k = gr.Slider(value=50, minimum=1, maximum=100, step=1, label="Top K")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    generate_button = gr.Button(value="Generate", label="Generate")

                with gr.Column():
                    gr.Markdown("## Outputs 🤖")
                    gr.Markdown("Contrastive Search generation:")
                    text_contrastive = gr.Textbox(value="", label="")
                    time_contrastive = gr.Number(value=0.0, precision=1, label="Generation time (ms)")
                    gr.Markdown("Top K Sampling generation:")
                    text_top_k = gr.Textbox(value="", label="")
                    time_top_k = gr.Number(value=0.0, precision=1, label="Generation time (ms)")

            # actions
            generate_button.click(
                fn=generate_top_k,
                inputs=[input_text, model_id, max_new_tokens, alpha, k, top_k, temperature, seed],
                outputs=[text_contrastive, time_contrastive, text_top_k, time_top_k]
            )

        with gr.TabItem("vs. Nucleus Sampling"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Inputs ✍️")
                    gr.Markdown("General options:")

                    #model_id = gr.Text(value="facebook/galactica-6.7b", label="Model Repository")
                    model_id = gr.Dropdown(choices= ["Alpha-7B1", "Alpha-1B7","Alpha-1B1" ], value="Alpha-1B7", label="Choosing LLM", show_label=True)

                    input_text = gr.Textbox(value="。", lines=5, label="Input Text")
                    max_new_tokens = gr.Slider(value=1024, minimum=100, maximum=2048, label="New tokens to generate")
                    gr.Markdown("Contrastive Search options:")
                    alpha = gr.Slider(value=0.6, minimum=0.01, maximum=1.0, step=0.01, label="Alpha")
                    k = gr.Slider(value=6, minimum=1, maximum=20, step=1, label="K")
                    gr.Markdown("Sampling options:")
                    top_p = gr.Slider(value=0.95, minimum=0.01, maximum=1.0, step=0.01, label="Top P")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    generate_button = gr.Button(value="Generate", label="Generate")

                with gr.Column():
                    gr.Markdown("## Outputs 🤖")
                    gr.Markdown("Contrastive Search generation:")
                    text_contrastive = gr.Textbox(value="", label="")
                    time_contrastive = gr.Number(value=0.0, precision=1, label="Generation time (ms)")
                    gr.Markdown("Nucleus Sampling generation:")
                    text_top_p = gr.Textbox(value="", label="")
                    time_top_p = gr.Number(value=0.0, precision=1, label="Generation time (ms)")

            # actions
            generate_button.click(
                fn=generate_top_p,
                inputs=[input_text, model_id, max_new_tokens, alpha, k, top_p, seed],
                outputs=[text_contrastive, time_contrastive, text_top_p, time_top_p]
            )

demo.launch(share=True)