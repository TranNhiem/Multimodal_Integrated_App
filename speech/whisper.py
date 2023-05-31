'''
@TranNhiem 2023/05/08

Implementation Whisper Model for ASR 

App Module Feature for Vietnamese & English Speech Recognition Module 

Reference Development Finetune Whisper Model 
https://huggingface.co/blog/fine-tune-whisper 

## Whisper Model 
Size	Layers	Width	Heads	Parameters	English-only	Multilingual
tiny	4	    384	    6	    39 M	    ✓	                    ✓
base	6	    512	    8	    74 M	    ✓	                    ✓
small	12	    768	    12	    244 M	    ✓	                    ✓
medium	24	    1024	16	    769 M	    ✓	                    ✓
large	32	    1280	20	    1550 M	    x	                    ✓


Pipeline 
1.
'''


#---------------------------------------------
### Audio Preprocessing 
#---------------------------------------------

'''
The Whisper feature extractor performs two operations:

1. Pads / truncates the audio inputs to 30s: any audio inputs shorter than 30s are padded to 30s with silence (zeros),
 and those longer that 30s are truncated to 30s
2. Converts the audio inputs to log-Mel spectrogram input features, a visual representation of the audio and the form of the input 
    expected by the Whisper model
'''
### Load WhisperFeatureExtractor
import os
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer
from datasets import Audio

model_name="openai/whisper-medium"

# ## 1 Loading Model 
weight_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/whisper/"
# Create the weight_path if it is not exist 
if not os.path.exists(weight_path): 
    os.makedirs(weight_path)

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=weight_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name, cache_dir=weight_path, language="vietnamese", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, cache_dir=weight_path, language="vietnamse", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")






