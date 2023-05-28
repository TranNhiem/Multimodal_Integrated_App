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


##-------------------------------------------------
## Langchain Section 
##-------------------------------------------------
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

template = """
The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.
 The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.



Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)


llm = OpenAI(model_name="gpt-3.5-turbo", # 'text-davinci-003'
             temperature=0.3, 
             max_tokens = 256)


# max_token_limit=40 - token limits needs transformers installed
memory= ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)

conversation_with_summary = ConversationChain(
    llm=llm, 
    memory=memory, 
    verbose=True, 
    prompt= prompt, 
)

def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    print("This is check input:", inp)
    output = conversation_with_summary.predict(input=inp)
    history.append((input, output))
    return history, history


block = gr.Blocks()


# with gr.Blocks() as demo:
#     gr.Markdown("""<h1><center>Assistant via SIF </center></h1>""")
#     chatbot = gr.Chatbot(label="Assistant")
#     message = gr.Textbox(show_label=False, placeholder="Enter your prompt and press enter", visible=True).style(container=False)
#     state = gr.State()
    
#     def clear_textbox(inputs, outputs):
#         inputs["message"] = ""  # Clear the message input
        
#     message.submit(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state], queue=False, on_submit=clear_textbox)
    

#     ## For Setting Hyperparameter 
#     with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
#         temperature = gr.Slider(
#             minimum=0.0,
#             maximum=1.0,
#             value=0.7,
#             step=0.1,
#             interactive=True,
#             label="Temperature",
#         )
#         top_p = gr.Slider(
#             minimum=0.0,
#             maximum=1.0,
#             value=1.0,
#             step=0.1,
#             interactive=True,
#             label="Top P",
#         )
#         max_output_tokens = gr.Slider(
#             minimum=16,
#             maximum=1024,
#             value=512,
#             step=64,
#             interactive=True,
#             label="Max output tokens",
#         )


# demo.queue().launch(debug = False)


with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>Assistant via SIF </center></h1>""")
    chatbot = gr.Chatbot(label="Assistant")
    message = gr.Textbox(show_label=False, placeholder="Enter your prompt and press enter", visible=True).style(container=False)
    state = gr.State()

    def clear_textbox(message, state):
        message.update("")  # Clear the message input
        
    def submit_callback(inputs, outputs):
        clear_textbox(inputs["message"], state)
        return chatgpt_clone(inputs, outputs)
    
    message.submit = submit_callback
    
demo.queue().launch(debug=True)


# with gr.Blocks() as demo:
    
   
#     chatbot = gr.Chatbot(value=[], elem_id="chatbot").style(height=650)
#     with gr.Row():
#         with gr.Column(scale=0.90):
#             txt = gr.Textbox(
#                 show_label=False,
#                 placeholder="Enter your prompt and press enter",
#             ).style(container=False) 
#         with gr.Column(scale=0.10):
#             cost_view = gr.Textbox(label='usage in $',value=0)

#     txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
#             generate_response, inputs =[chatbot,],outputs = chatbot,).then(
#             calc_cost, outputs=cost_view)
            
# demo.queue()

# print(conversation_with_summary.predict(input="Hi there! I want to ask you a question about how to write a simple transformer model in python for computer vision"))

# print(conversation_with_summary.predict(input=" This Model i will use for image classification tasks"))

# print(conversation_with_summary.predict(input=" I also want to create the lightweight of this model enable to run on mobile devices"))

# print(conversation_with_summary.predict(input=" can you write an example python snipe code for this"))
