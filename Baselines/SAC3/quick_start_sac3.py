# THIS IS OUR IMPLEMENTATION OF SAC3, WITH OFFICIAL CODE USED FROM THIER REPOSITORY https://github.com/intuit/sac3.
# Necessary libraries and environment setup is same as provided in their work.
# Setup the environment and local the LLM locally (Qwen2.5, LLaMA3.1 etc), and use the input pkl file in the mentioned format. 
# This can be directly setup and run using "python filename.py". Experiments were run on single NVIDIA A6000, A100 GPUs (others are possible too).

import os
# update your cache dir 
# os.environ['HUGGINGFACE_HUB_CACHE'] = '/custom/path/to/hf_cache'
# os.environ['HF_HOME'] = '/custom/path/to/hf_cache'

import torch
from peft import PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
import os
import re

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = 'cuda'
model_pth = "Qwen/Qwen2.5-7B-Instruct"
cache_dir = "/custom/path/to/qwen2.5"
tokenizer = AutoTokenizer.from_pretrained(model_pth, cache_dir=cache_dir, padding = True)
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_pth, cache_dir=cache_dir,trust_remote_code=True,output_attentions=True, torch_dtype="auto")
# model = model.to(device)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", cache_dir=cache_dir, trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device)
tokenizer.pad_token_id = tokenizer.bos_token_id
tokenizer.padding_side = "left"
def call_guanaco_33b(prompt, max_new_tokens):
    # 16 float 
    model_name = "huggyllama/llama-30b"
    adapters_name = 'timdettmers/guanaco-33b'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # offload_folder="/custom/path/to/SageMaker/hf_cache",
        max_memory= {i: '16384MB' for i in range(torch.cuda.device_count())}, # V100 16GB
    )
    model = PeftModel.from_pretrained(model, adapters_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # prompt 
    formatted_prompt = (
        f"A chat between a curious human and an artificial intelligence assistant."
        f"The assistant gives helpful, concise, and polite answers to the user's questions.\n"
        f"### Human: {prompt} ### Assistant:"
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=max_new_tokens)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    res_sp = res.split('###')
    output = res_sp[0] + res_sp[2]
    
    return output 
    
    
def call_falcon_7b(prompt, max_new_tokens):
    # 16 float     
    model = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    sequences = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        res = seq['generated_text']

    return res

def call_mistral_7b(prompt, max_new_tokens):
    # formatted_prompt = f"[INST]\n You are a artificial intelligence assistant. Give helpful, concise and polite answers for the human questions.\n Human:{prompt} \n Assistant: \n[\INST]"
    # formatted_prompt = (
    #     f"A chat between a curious human and an artificial intelligence assistant."
    #     f"The assistant gives helpful, concise, and polite answers to the user's questions.\n"
    #     f"### Human: {prompt} ### Assistant:"
    # )
    formatted_prompt = f"{prompt}"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    N = inputs['input_ids'].shape[1]
    inputs = inputs.to(device)
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=100)
    gen_op = tokenizer.batch_decode(generate_ids[:,N:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print("Output: ")
    
    return gen_op

def call_llama2_7b(prompt, max_new_tokens):
    # formatted_prompt = (
    #     f"A chat between a curious human and an artificial intelligence assistant."
    #     f"The assistant gives helpful, concise, and polite answers to the user's questions.\n"
    #     f"### Human: {prompt} ### Assistant:"
    # )
    formatted_prompt = f"[INST]\n You are a artificial intelligence assistant. Give helpful, concise and polite answers for the human questions.\n Human:{prompt} \n Assistant: \n[\INST]"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    N = inputs['input_ids'].shape[1]
    inputs = inputs.to(device)
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=500)
    gen_op = tokenizer.batch_decode(generate_ids[:,N:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return gen_op

def paraphrase(question, number, model, temperature):
    '''
    Inputs: 
    question - original user query
    number - how many perturbed questions
    model - GPTs or open-sourced models
    temperature - typically we use 0 here 

    Output:
    perb_questions - perturbed questions that are semantically equivalent to the question
    '''

    perb_questions = []  
    prompt_temp = f'For question Q, provide {number} semantically equivalent questions.'
    # prompt = prompt_temp + '\nQ: ' + question
    # prompt = '[INST]\n' + prompt_temp + '\nQ:' + question + '\n[\INST]\n\n'
    # prompt = '<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n'+ prompt_temp + '\nQ:' + question + '<|end|>\n<|assistant|>'
    # prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|>"+prompt_temp+"\nQ:" +question+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    prompt = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n"+prompt_temp+"\nQ:" + question +' <|im_end|>\n<|im_start|>assistant\n'

    #res = llm_models.call_openai_model(prompt, model, temperature) # openai model call 
    res = call_mistral_7b(prompt, model)
    # print(res)
    lines = res.split('\n')
    # print(lines)

    for line in lines:
        if len(perb_questions) < 5:  # Stop after collecting 5 questions
            question = re.sub(r'^\d+\.\s+', '', line).strip()
            if question:
                perb_questions.append(question)
        else:
            break
    # print(perb_questions)
    # res = call_llama2_7b(prompt, model)
    # res_split = res.split('\n')
    # for i in range(len(res_split)):
    #     perb_questions.append(res_split[i])
    
    return perb_questions

class Evaluate:
    def __init__(self, model):
        self.model = model
        # self.prompt_temp = 'Answer the following question:\n'
        # self.prompt_temp = '[INST]\nAnswer the following question:'
        # self.prompt_temp = '<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>'
        # self.prompt_temp = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>"
        self.prompt_temp = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n"

    def mistral_parallel(self, prompt, temperature):
        res = call_mistral_7b(prompt, max_new_tokens=100) # openai model call
        return res

    def llama2_parallel(self, prompt, temperature):
        res = call_llama2_7b(prompt, max_new_tokens=100) # openai model call
        return res 
    
    def self_evaluate_api(self, self_question, temperature, self_num):

        # prompt = self.prompt_temp + '\nQ:' + self_question    #for Mistral and LLaMa2 
        prompt = self.prompt_temp + '\n' + self_question      #for Phi3
        self_responses = []
        with ThreadPoolExecutor(max_workers=self_num+2) as executor:
            futures = [executor.submit(self.mistral_parallel, prompt, temperature) for _ in range(self_num)]
            for future in concurrent.futures.as_completed(futures):
                self_responses.append(future.result())

        return self_responses
    
    def perb_evaluate_api(self, perb_questions, temperature):
        perb_responses = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # futures = [executor.submit(self.mistral_parallel, self.prompt_temp + '\nQ:' + perb_question + '\n[\INST]\n', temperature) for perb_question in perb_questions]
            # futures = [executor.submit(self.mistral_parallel, self.prompt_temp + '\nQ:' + perb_question + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n', temperature) for perb_question in perb_questions]
            futures = [executor.submit(self.mistral_parallel, self.prompt_temp + '\nQ:' + perb_question + '<|im_end|>\n<|im_start|>assistant\n', temperature) for perb_question in perb_questions]
            for future in concurrent.futures.as_completed(futures):
                perb_responses.append(future.result())
            # print(perb_responses)
        return perb_responses
    
    def self_evaluate(self, self_question, temperature, self_num):
        '''
        Inputs: 
        self_question - original user query 
        temperature - [0,1] for LLM randomness
        self_num - how many generated responses given this question

        Outputs:
        self_responses - generated responses given this question with different temperatures
        '''

        self_responses = []  
        prompt = self.prompt_temp + '\nQ:' + self_question
    
        for i in range(self_num):  
            # llm model: GPTs, open-source models (falcon, guanaco)
            #if self.model in ['gpt-3.5-turbo','gpt-4']:
            #    res = call_openai_model(prompt, self.model, temperature) # openai model call
            if self.model == 'guanaco-33b':
                res = call_guanaco_33b(prompt, max_new_tokens = 200)
            elif self.model == 'falcon-7b':
                res = call_falcon_7b(prompt, max_new_tokens = 200)
            elif self.model == 'mistral-7b':
                res = call_mistral_7b(prompt, max_new_tokens = 200)
            # other open-sourced llms 
            self_responses.append(res)

        return self_responses
    
    def perb_evaluate(self, perb_questions, temperature):
        '''
        Inputs: 
        perb_questions - perturbed questions that are semantically equivalent to the original question
        temperature - [0,1] for LLM randomness

        Outputs:
        perb_responses - generated responses given the perturbed questions
        '''
        
        perb_responses = []   
        for i in range(len(perb_questions)):
            prompt = self.prompt_temp + '\nQ:' + perb_questions[i]
            # llm model: GPTs, open-source models (falcon, guanaco)
            #if self.model in ['gpt-3.5-turbo','gpt-4']:
            #    res = call_openai_model(prompt, self.model, temperature) # openai model call 
            if self.model == 'guanaco-33b':
                res = call_guanaco_33b(prompt, max_new_tokens = 200)
            elif self.model == 'falcon-7b':
                res = call_falcon_7b(prompt, max_new_tokens = 200)
            elif self.model == 'mistral-7b':
                res = call_mistral_7b(prompt, max_new_tokens = 200)
            # other open-sourced llms 
            perb_responses.append(res)
  
        return perb_responses

class SemanticConsistnecyCheck:
    def __init__(self, model):
        self.model = model
        # self.prompt_temp = """
        # Are the following two Question-Answer(QA) pairs semantically equivalent? 
        # Provide your best guess and the probability that it is correct (0.0 to 1.0).
        # Given ONLY the guess (Yes or No) and probability, no other words or explanation. 
        # For example:
        # Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!> 
        # Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; 
        # just the probability!>
        # """
        # self.prompt_temp = '<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n' + "There are two pairs of question-answers below, first pair and second pair. Are the pairs semantically equivalent? Generate the Output as only Yes or No if they are similar. Do not generate anything else."
        # self.prompt_temp = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n" + "There are two pairs of question-answers below, first pair and second pair. Are the pairs semantically equivalent? Generate the Output as only Yes or No if they are similar. Do not generate anything else."
        # self.prompt_temp = "There are two pairs of question-answers below, first pair and second pair. Are the pairs semantically equivalent? Generate the Output as only Yes or No if they are similar. Do not generate anything else."
        self.prompt_temp = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n" + "There are two pairs of question-answers below, first pair and second pair. Are the pairs semantically equivalent? Generate the Output as only Yes or No if they are similar. Do not generate anything else."
    def mistral_parallel(self, prompt, temperature):
        res = call_mistral_7b(prompt, max_new_tokens=20) 
        return res
    
    def llama2_parallel(self, prompt, temperature):
        res = call_llama2_7b(prompt, max_new_tokens=100) 
        return res
    
    def score_scc_api(self, question, target_answer, candidate_answers, temperature):
        if target_answer is None:
            raise ValueError("Target answer cannot be None. ")

        sc_output = []  
        target_pair = 'Q:' + question + '\nA:' + target_answer
        num_candidate_answer = len(candidate_answers)

        with ThreadPoolExecutor(max_workers=num_candidate_answer+2) as executor:
            all_res = []
            for i in range(num_candidate_answer):
                candidate_pair = 'Q:' + question + '\nA:' + candidate_answers[i]
                # prompt = '[INST]\n' + self.prompt_temp + '\nThe first QA pair is:\n' + target_pair + '\nThe second QA pair is:\n' + candidate_pair + '\nOutput:'+ '\n[\INST]\n\n'
                prompt = '[INST]\n' + self.prompt_temp + '\nFirst pair:\n' + target_pair + '\nSecond pair:\n' + candidate_pair + '\nOutput (Yes or No):'+ '\n[\INST]\n'
                output = executor.submit(self.mistral_parallel, prompt, temperature)
                # print(output)
                all_res.append(output)
                print(all_res)

            for future in concurrent.futures.as_completed(all_res):
                try:
                    res = future.result()
                    # print(res)
                    # print('count')
                    guess = res.split(':')[1].split('\n')[0].strip()
                except Exception as e:
                    print(f"Error occurred in score_scc_api: {e}. Using placeholder value.")
                    guess = "E"
                value = 0 if guess == 'Yes' else 1
                sc_output.append(value)
            
        score = sum(sc_output)/num_candidate_answer
        return score, sc_output
        
    def score_scc(self, question, target_answer, candidate_answers, temperature):
        '''
        Inputs:
        question - original user query
        target_answer - generated response given the original question (temp=0) if not provided by user 
        candidate_answers - generated responses given the question (original + perturbed)
        temperature - [0,1] for LLM randomness

        Outputs:
        score - inconsistency score (hallucination metric) 
        sc_output - specific score for each candidate answers compared with the target answer  
        '''

        if target_answer is None:
            raise ValueError("Target answer cannot be None. ")

        sc_output = []  
        e_index = []
        target_pair = 'Q:' + question + '\nA:' + target_answer
        num_candidate_answer = len(candidate_answers)
        for i in range(num_candidate_answer): 
            candidate_pair = 'Q:' + question + '\nA:' + candidate_answers[i]
            # prompt = self.prompt_temp + '\nThe first QA pair is:\n' + target_pair + '\nThe second QA pair is:\n' + candidate_pair
            # prompt = self.prompt_temp + '\nFirst pair:\n' + target_pair + '\nSecond pair:\n' + candidate_pair + '\nOutput(Yes or No):<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n' 
            prompt = self.prompt_temp + '\nFirst pair:\n' + target_pair + '\nSecond pair:\n' + candidate_pair + '\nOutput(Yes or No):\n<|im_end|>\n<|im_start|>assistant\n'
            #res = llm_models.call_openai_model(prompt, self.model, temperature) # openai model call 
            res = call_mistral_7b(prompt, self.model)
            # print(res)
            print('Count is', i)
            # try:
            #     guess = res.split(':')[1].split('\n')[0].strip()
            # except Exception as e:
                # print(f"Error occurred in score_scc: {e}. Using placeholder value.")
            #     e_index.append(i)
            #     guess = "E"
            try:
                if re.search(r'\bYes\b', res, re.IGNORECASE) and not re.search(r'\(Yes or No\)', res, re.IGNORECASE):
                    guess = "Yes"
                else:
                    guess = "No"  # Default to 'No' if 'Yes' is not found
            except Exception as e:
                print(f"Error occurred in score_scc: {e}. Using placeholder value.")
                e_index.append(i)
                guess = "E"
            # print(res, guess)
            print(guess)
            value = 0 if guess == 'Yes' else 1
            # print('value',value)
            sc_output.append(value)
        if num_candidate_answer > 0:
            score = sum(sc_output)/num_candidate_answer
        else:
            score = 1
        return score, sc_output
    
import pickle
with open('/custom/path/to/qwen_hotpot.pkl', 'rb') as f:
#The pickle file should be in this list format -> [["Q1", "A1"], ["Q2","A2"], ......, ["Qn","An"]], where each Ai corresponds to the original output answer to the dataset Qi.
#The example filename qwen_hotpot.pkl indicates it contains Q-A pairs consisting of Questions from HotpotQA (validation/test set) and Answers generated by Qwen2.5-7b-Instruct.    
    answers = pickle.load(f)
sc2_list = []
sac3_q_list = []
for i in tqdm(range(len(answers))): 
    question = answers[i][0]
    print("Question: ", question)
    target_answer = answers[i][1]
    print(f"Current iteration is : {i}.")
    # question pertubation
    gen_question = paraphrase(question, number = 5, model = 'mistral-7b', temperature=1.0)

    # llm evaluation
    llm_evaluate = Evaluate(model='mistral-7b')
    #self_responses = llm_evaluate.self_evaluate_api(self_question = question, temperature = 1.0, self_num = 2)
    perb_responses = llm_evaluate.perb_evaluate_api(perb_questions = gen_question, temperature=1.0)
    scc = SemanticConsistnecyCheck(model='mistral-7b')
    #sc2_score, sc2_vote = scc.score_scc_api(question, target_answer, candidate_answers = self_responses, temperature = 0.0)

    sac3_q_score, sac3_q_vote = scc.score_scc(question, target_answer, candidate_answers = perb_responses, temperature = 1.0)
    sac3_q_list.append(sac3_q_score)
    with open('qwen_sac3_hotpot.pkl', 'wb') as f:
        pickle.dump(sac3_q_list, f)
