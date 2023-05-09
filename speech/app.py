import torch
import os
import gradio as gr
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig


model_name ="openai/whisper-large" ##"joey234/whisper-medium-vi" --> This model finetune for Vietnamese Dataset 

language = "Vietnamese"
task = "transcribe"
# ## 1 Loading Model 
weight_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/whisper/"
# Create the weight_path if it is not exist 
if not os.path.exists(weight_path): 
    os.makedirs(weight_path)

model = WhisperForConditionalGeneration.from_pretrained(model_name,load_in_8bit=True, cache_dir= weight_path,  device_map="auto")#
tokenizer = WhisperTokenizer.from_pretrained(model_name, cache_dir= weight_path,language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name,cache_dir= weight_path, language=language, task=task)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)


def transcribe(audio):
    # with torch.cuda.amp.autocast(torch.float16):
    with torch.autocast(device_type='cuda'):
        text = pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title=" Whisper ASR Test",
    description="Real time speech recognition Test",
)

iface.launch(share=True)