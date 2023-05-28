'''
@TranNhiem 2023/05
This design including 2 Sections:

1. Using The Pay API LLM Model 
    + OpenAI API (gpt-3.5-turbo) & GPT-3 API (text-davinci-003)
    
2. Using Open-Source Pretrained Language Model (Self-Instructed FineTune Model) 
    + BLOOMZ 
    + LLaMA
    + Falcon
    + MPT 

3. Self-Instruct Finetune Model on Different Dataset 
    + Alpaca Instruction Style  
    + Share GPT Conversation Style 
    + Domain Target Instruction Style 

4 Pipeline Development 

1.. FineTune Instruction LLM  --> 2.. Langain Memory System  --> Specific Design Application Domain 

    4.1 Indexing LLM (Augmented Retrieved Documents )
    4.2 Agent LLM (Design & Invent New thing for Human)

'''

import os 
import openai
import gradio as gr
## Setting OpenAI API 


##-------------------------------------------------
## Using OpenAI API and Langchain 
##-------------------------------------------------
API_TYPE = "azure"
API_BASE = "https://sslgroupservice.openai.azure.com/"
API_VERSION = "2023-03-15-preview" #"2022-06-01-preview"#"2023-03-15-preview"
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-35-turbo"#"gpt-3.5-turbo" #"gpt-35-turbo" for Azure API, OpenAI API "gpt-3.5-turbo"#"gpt-4", "text-davinci-003"

# Set up API
def setup_api(api="azure"):
    if api == "azure":
        openai.api_type = API_TYPE
        openai.api_base = API_BASE
        openai.api_version = API_VERSION
        openai.api_key = API_KEY
    else:
        openai.organization = "org-PVVobcsgsTm9RT8Ez5DubzbX" # Central IT account
        #openai.api_key = API_KEY
        openai.api_key = os.getenv("OPENAI_API_KEY")

setup_api(api="openAI") #azure



## -------------------Setting Langchain Section =-------------------
## Setting LangChain Summary&BufferMemory 
'''
2 Advanced Setting Memory 
    2.1 Summary+ Buffer Memory
    2.2 Knowledge Graph Memory
'''
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain 
from langchain import OpenAI
from langchain.prompts.prompt import PromptTemplate


# Setting Prompt Template
#The following is a friendly conversation between a human and an AI. The AI is talkative and .
template = """
The following is a conversation between a human and AI assistant. The AI assistant is helpful, creative, clever, and very friendly, provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.
The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:
{history}

Conversation:
Human: {input}
AI:

"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)

with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>Assistant via SIF </center></h1>""")
    chatbot = gr.Chatbot(label="Assistant").style(height=500)
    
    with gr.Row():
        message = gr.Textbox(show_label=False, placeholder="Enter your prompt and press enter", visible=True)
    state = gr.State()
    # max_token_limit=40 - token limits needs transformers installed
    memory= ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)

    def respond(message, chat_history, temperature=0.7, top_p=1.0, max_output_tokens=512):
        
        llm = OpenAI(model_name="gpt-3.5-turbo", # 'text-davinci-003'
             temperature=temperature, 
             top_p=top_p,
             max_tokens = max_output_tokens)
        
        conversation_with_summary = ConversationChain(
            llm=llm, 
            memory=memory, 
            verbose=True, 
            prompt= prompt, 
        )
        bot_message = conversation_with_summary.predict(input=message)
        message= "👤: "+ message
        bot_message= "😃: "+ bot_message
        chat_history.append((message, bot_message))
        #time.sleep(1)
        return "", chat_history
     
    ## For Setting Hyperparameter 
    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    message.submit(respond, inputs=[message, chatbot, temperature, top_p, max_output_tokens], outputs=[message, chatbot], queue=False, )
    
# demo.queue()
# demo.launch()



## ----------------------------For Testing The system Memory ----------------------------

# print(conversation_with_summary.predict(input="Hi there! I want to ask you a question about how to write a simple transformer model in python for computer vision"))

# print(conversation_with_summary.predict(input=" This Model i will use for image classification tasks"))

# print(conversation_with_summary.predict(input=" I also want to create the lightweight of this model enable to run on mobile devices"))

# print(conversation_with_summary.predict(input=" can you write an example python snipe code for this"))



##-----------------------------------------------------------
## Using Huggingface API and Langchain
##-----------------------------------------------------------

from transformers import  AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain

weight_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/BLOOMZ/"

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-3b", cache_dir=weight_path)
base_model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloomz-3b",
    # load_in_8bit=True,
    device_map='auto',
    cache_dir=weight_path,)

## Add the model the base Configurations to manipulate the Model Decoding Method 
local_llm = HuggingFacePipeline(pipeline=pipe)

template = 

"""
The conversation between a human input and an AI assistant follows a specific pattern. The human provides an instruction or request that describes a task or action they would like the AI assistant to perform. The AI assistant then generates a response that appropriately completes the request or fulfills the given instruction.

Instruction: {instruction}
Output Response:

"""

prompt_template = PromptTemplate(template=template, input_variables=["instruction"])


#memory= ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)
memory= ConversationSummaryBufferMemory(llm=local_llm, max_token_limit=40)

conversation_with_summary = ConversationChain(
            llm=local_llm, 
            memory=memory, 
            verbose=True, 
            prompt= prompt_template, 
        )

print(conversation_with_summary.predict(input="Hi there! I want to ask you a question about how to write a simple transformer model in python for computer vision"))

print(conversation_with_summary.predict(input=" This Model i will use for image classification tasks"))

print(conversation_with_summary.predict(input=" I also want to create the lightweight of this model enable to run on mobile devices"))

print(conversation_with_summary.predict(input=" can you write an example python snipe code for this"))

