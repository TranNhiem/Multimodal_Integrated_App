'''
TranNhiem 2023/05/31 

The Pipeline of Crate a Diversity of Instruction & High Quality dataset Covering a Wide Range of Human Tasks

1. Synthesize a Diversity of Instruction from LLM via Initialize 175 human tasks
    + This based The alpaca Instruction dataset data generation Pipeline 
    
    + An instruction finetune dataset derived from Alpaca-52K, using the evolution method
    https://huggingface.co/datasets/victor123/evol_instruct_70k 
    
2. Human and LLM interaction Dataset ShareGPT 
    + Using the High Quality Filter ShareGPT Dataset 
    ShareGPT_Vicuna_unfiltered
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered 
    conversations collected from ShareGPT, with a specific focus on customized creative conversation.
    https://huggingface.co/datasets/RyokoAI/ShareGPT52K 

3. dataset of its kind and contains 10k instances with safety preferences.
    dataset of its kind and contains 10k instances with safety preferences.
    + https://github.com/PKU-Alignment/safe-rlhf#pku-saferlhf-dataset

4. Dataset cover Q&A, Dialogue, and Instruction
    A combination of some subsets of OIG, P3 and Stackoverflow. Covers topics like general QA, customized creative questions.
    https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations 

5. Dataset all Create by Human context information 
    + Dolly 

    + OpenAssistant Dataset 

6. 


### Reference Dataset 
https://github.com/OpenBMB/ToolBench 

'''