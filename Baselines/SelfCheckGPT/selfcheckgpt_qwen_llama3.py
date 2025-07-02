# THIS IS OUR IMPLEMENTATION OF SeflCheckGPT-MQAG, WITH OFFICIAL CODE USED FROM THIER REPOSITORY https://github.com/potsawee/selfcheckgpt.
# Necessary libraries and environment setup is same as provided in their work.
# Setup the environment and local the LLM locally (Qwen2.5, LLaMA3.1 etc), and use the input pkl file in the mentioned format. 
# This can be directly setup and run using "python filename.py". Experiments were run on single NVIDIA A6000, A100 GPUs (others are possible too).

import torch
import spacy
import re
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore
import pickle
from tqdm import tqdm
import traceback

with open('/custom/path/to/qwen_hotpot.pkl', 'rb') as f:
    answers_cleaned = pickle.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from transformers import   LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# #For LLaMA3.1-8B
# cache_dir = '/custom/path/to/llama31-8b'
# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", cache_dir=cache_dir, trust_remote_code=True, torch_dtype=torch.float16,device_map="auto")

#For Qwen2.5-7B
model_pth = "Qwen/Qwen2.5-7B-Instruct"
cache_dir = "/custom/path/to/qwen2.5"
tokenizer = AutoTokenizer.from_pretrained(model_pth, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_pth, cache_dir=cache_dir,trust_remote_code=True,output_attentions=True, torch_dtype="auto")
model = model.to(device)
# tokenizer.pad_token_id=tokenizer.bos_token_id
# tokenizer.padding_side="left"
# template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a Question-answering assistant, only answer the question.<|eot_id|><|start_header_id|>user<|end_header_id|>\nQuestion: <question> <|eot_id|><|start_header_id|>assistant<|end_header_id|>"  #For LLaMA3.1
template = '<|im_start|>system\nYou are a Question-answering assistant, only answer the question.<|im_end|>\n<|im_start|>user\nQuestion: <question> <|im_end|>\n<|im_start|>assistant\n'  #For Qwen2.5
selfcheck_mqag = SelfCheckMQAG(device=device) 
# selfcheck_bertscore = SelfCheckBERTScore()
nlp = spacy.load("en_core_web_sm")
mqag_score = []
bert_score = []
for question, answer_temp0 in tqdm(answers_cleaned[6000:]):
    answer_temp0 = answer_temp0.replace("</s>", "")
    # print(answer_temp0)
    # answer_temp0 = answer_temp0[1]
    prompt = [template.replace("<question>", question)]*10
    tokenized_inputs = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding = "longest", return_attention_mask = True)
    tokenized_inputs = tokenized_inputs.to(device)
    N=tokenized_inputs['input_ids'].shape[1]
    outputs = model.generate(**tokenized_inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens = 50, do_sample = True, temperature = 1)
    predicted_token_ids = outputs['sequences']
    decoded_output = tokenizer.batch_decode(predicted_token_ids[:, N:], skip_special_tokens=True)
    # print(decoded_output)
    doc = nlp(answer_temp0)
    # sentences = [sent.text[:-4] if sent.text[-4:] == "</s>" else sent.text for sent in doc.sents]
    sentences = [sent.text for sent in doc.sents]
    # print(sentences)
    print("SelfCheck running on {} sentences...".format(len(sentences)))
    try: 
        sent_scores_mqag = selfcheck_mqag.predict(
        sentences,
        answer_temp0,
        decoded_output,
        num_questions_per_sent = 5,
        scoring_method = 'bayes_with_alpha',
        beta1 = 0.95, beta2 = 0.95,
        )
        print(f"MQAG Score for question: {question}", answer_temp0, ": ", sent_scores_mqag)
    except Exception as e:
        print("Exception: ", e)
        # print(traceback.format_exc())
        sent_scores_mqag = []
    # sent_scores_mqag = selfcheck_mqag.predict(
    #     sentences,
    #     answer_temp0,
    #     decoded_output,
    #     num_questions_per_sent = 5,
    #     scoring_method = 'bayes_with_alpha',
    #     beta1 = 0.95, beta2 = 0.95,
    #     )
    # print(f"MQAG Score for question: {question}", answer_temp0, ": ", sent_scores_mqag)

    # for s1 in sent_scores_mqag:
    #    print("{:.4f}".format(s1))
    mqag_score.append([question, answer_temp0, sent_scores_mqag])
    file_path = '/custom/path/to/qwen_selfcheck_hotpot.pkl'
    # Save the list of lists to the specified file using pickle
    with open(file_path, 'wb') as file:
        pickle.dump(mqag_score, file)
    