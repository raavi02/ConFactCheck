#Import libraries
import spacy
from scipy.stats import kstest
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline, AutoModelWithLMHead
import torch
import pickle
from tqdm import tqdm
import stanza
import re
import argparse
import random
from datasets import load_dataset
import ast

from openai import OpenAI

client = OpenAI(api_key = "<INSERT-API-KEY>")
# import en_core_web_sm

#Import NER, POS taggers
stanza.download('en')     
nlp_ner = stanza.Pipeline(lang='en', processors='tokenize,ner')
nlp_pos = stanza.Pipeline(lang='en', processors='tokenize,pos')
nlp_coreref = stanza.Pipeline("en", processors="tokenize,coref")
nlp_sent = spacy.load("en_core_web_sm")
# nlp_sent = en_core_web_sm.load()




parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, help='Device to run the code on')
parser.add_argument('--method', type=str, help='Method: ner/pos/random')
parser.add_argument('--model_name', type=str, help='Name of the model used to generate the original sentence')
parser.add_argument('--model_cached_pth', type=str, help='Path to load the cached model from')
parser.add_argument('--sentences', type=str, help='Pickle file for hallucinated sentences')
parser.add_argument('--metrics_file', type=str, help='Path to store the metrics at')

args = parser.parse_args()


device = args.device
model_pth = args.model_name
cache_dir = args.model_cached_pth
method = args.method

#--------------------Importing the Question generation model-------------------------------------------------------------
tokenizer_ques_gen = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model_ques_gen = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap", device_map=device)


#-------------------Importing the user model-----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_pth, cache_dir=cache_dir, padding = True)
model = AutoModelForCausalLM.from_pretrained(model_pth, cache_dir=cache_dir, trust_remote_code=True,output_attentions=True, device_map=device)
# tokenizer.pad_token_id = tokenizer.bos_token_id
# tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token # for Qwen2.5

qa_model = pipeline("question-answering")

def extract_sentences(orig_answer):
    doc = nlp_sent(orig_answer)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def do_coreference(sentence):
    corerefernced_Sent = nlp_coreref(sentence)
    for sent in corerefernced_Sent.sentences:
        for word in sent.words:
            try:
                i = 0
                while(i < len(word.coref_chains) and word.text in word.coref_chains[i].chain.representative_text):
                    i = i + 1
                if i != len(word.coref_chains):
                    sentence = sentence.replace(word.text, word.coref_chains[i].chain.representative_text)
            except:
                i = i + 1
                pass
    return sentence

def generate_questions(context):
    inputs = tokenizer_ques_gen(context, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model_ques_gen.generate(**inputs, max_length=200, do_sample=True, num_return_sequences = 2, temperature = 0.5)
    question_answer_pairs = tokenizer_ques_gen.batch_decode(outputs, skip_special_tokens=False)
    question_answer_list = [question_answer.replace(tokenizer_ques_gen.pad_token, "").replace(tokenizer_ques_gen.eos_token, "") for question_answer in question_answer_pairs]
    generated_question_answer_list = [question_answer.split(tokenizer_ques_gen.sep_token) for question_answer in question_answer_list]
    return generated_question_answer_list

def extract_text_between_double_quotes(input_string):
    #Function to extract text between double quotes
    pattern = r'"([^"]*)"'
    try:
        matches = re.findall(pattern, input_string)
    except:
        matches = []
    return matches

def generate_questions_based_on_factual_parts(sentence, method='ner'):
    #Extracting atomic facts
    def get_question(answer, context, max_length=64):
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = tokenizer_ques_gen([input_text], return_tensors='pt')
        features = features.to(device)
        output = model_ques_gen.generate(input_ids=features['input_ids'], 
                    attention_mask=features['attention_mask'],
                    max_length=max_length)
        ques = tokenizer_ques_gen.decode(output[0])
        return ques
    if method == 'pos':
        double_quote_words = extract_text_between_double_quotes(sentence)
        text = sentence
        try:
            for i, double_quote_word in zip(range(len(double_quote_words)), double_quote_words):
                # print(double_quote_word)
                text = text.replace('"{}"'.format(double_quote_word), "DOUBLEQUOTES" + str(i))
        except:
            pass
        doc = nlp_pos(text)
        is_factual = []
        split_text = []
        for sent in doc.sentences:
            for word in sent.words:
                split_text.append(word.text)
                if word.xpos == "NNP" or word.xpos == "NNPS" or word.xpos == "CD" or word.xpos == "RB":
                    is_factual.append(1)
                elif word.upos == "PUNCT":
                    is_factual.append(2)
                elif word.xpos == "IN":
                    is_factual.append(3)
                else: is_factual.append(0)
        i = 0
        atomic_facts = []
        while (i < len(is_factual)):
            s = ""
            while i < len(is_factual) and (is_factual[i] ==1 or (is_factual[i] == 2 and is_factual[i-1]!=0  and i < (len(is_factual) - 1) and is_factual[i+1] !=0) or (is_factual[i] == 3 and is_factual[i-1]!=0  and i < (len(is_factual) - 1) and is_factual[i+1] !=0)):
                s += split_text[i] + " "
                i +=1
            if s != "":
                atomic_facts.append(s)
            i += 1
        atomic_facts = [fact[:-1] for fact in atomic_facts]
        # print(atomic_facts)
        output_list = []
        for element in atomic_facts:
            if "DOUBLEQUOTES" in element:
                # Extract the integer after "DOUBLEQUOTES"
                index = int(element.split("DOUBLEQUOTES")[1].strip())
                
                # Replace with the corresponding element from double_quote_words
                output_list.append(double_quote_words[index])
            else:
                output_list.append(element)
    if method == 'ner':
        ner_sent = nlp_ner(sentence)
        output_list = [ent.text for sent in ner_sent.sentences for ent in sent.ents]
    if method == 'random':
        ner_sent = nlp_ner(sentence)
        num_random_facts = len([ent.text for sent in ner_sent.sentences for ent in sent.ents])
        random_facts_indices = random.sample(range(0, len(sentence.split())), num_random_facts)
        output_list = [sentence.split()[index] for index in random_facts_indices]




    print(method," ", output_list)
    questions_answer_list = []
    pattern = r'<pad> question: (.+?)</s>'
    # print("Atomic facts", output_list)
    for atomic_fact in output_list:
        gen_ques = get_question(atomic_fact, sentence)
        gen_ques = re.search(pattern, gen_ques).group(1)
        # if output_list.any() in gen_ques:
        questions_answer_list.append([gen_ques, atomic_fact])
        # print(questions_answer_list)
    return questions_answer_list

def build_robust_prompt(extracted_facts, regenerated_facts):
    if not extracted_facts:
        return "Output: [1]"
    assert len(extracted_facts) == len(regenerated_facts)
    pairs = "\n".join(
        f"{i+1}. Extracted: \"{ex}\"\n   Regenerated: \"{re}\""
        for i, (ex, re) in enumerate(zip(extracted_facts, regenerated_facts))
    )
    n = len(extracted_facts)

    return f"""You are a fact comparison expert. Your task is to determine whether pairs of extracted and regenerated facts refer to the same real-world entity, concept, or meaning.

For each pair:
- Return `0` if the two facts **refer to the same thing**, even if the wording, specificity, or structure is different.
- Return `1` if the two facts **do not refer to the same thing**, or if their meanings conflict.

### Consider the following when making your decision:
- Minor differences in wording, grammar, or capitalization should be ignored.
- Partial vs full entity names (e.g., "Vancouver" vs "Vancouver, British Columbia") should count as a match if referring to the same entity.
- Aliases and synonymous expressions (e.g., "Roger Pirates" vs "Roger crew") should count as a match.
- Abbreviations and full forms (e.g., "UCLA" vs "University of California, Los Angeles") are considered matches.
- Only return `1` when the two facts clearly refer to different entities, times, concepts, or are ambiguous enough to be unrelated.

### Format:
Return a Python-style list of exactly {n} binary values (0 or 1), corresponding to each fact pair in order.
**Do not output anything else.**
If unsure, make your best judgment and still return a complete list.

### Examples:
Example A:
Extracted: "President Donald J. Trump"
Regenerated: "Donald Trump"
→ 0

Example B:
Extracted: "Vancouver, British Columbia."
Regenerated: "Vancouver"
→ 0

Example C:
Extracted: "five"
Regenerated: "5 seasons"
→ 0

Example D:
Extracted: "UCLA"
Regenerated: "University of California, Los Angeles"
→ 0

Example E:
Extracted: "Microsoft"
Regenerated: "Apple"
→ 1

Now judge the following fact pairs:

{pairs}

Output:"""

def call_openai_model(prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.0,
        max_output_tokens=100
        )
    output_str = response.output[0].content[0].text.strip()

    # Convert string like "[0, 0, 1]" to actual Python list
    try:
        output_list = ast.literal_eval(output_str)
        if isinstance(output_list, list) and all(x in [0, 1] for x in output_list):
            return output_list
        else:
            raise ValueError("Output is not a valid list of 0s and 1s.")
    except Exception as e:
        print(f"Error parsing model output: {e}")
        return []

def generate_pinpointed_answers(generated_question_answer_list, device):
    # prompt = [f"<s>[INST] Question: {question_answer[0]}\n Answer with reasoning: [/INST]" for question_answer in generated_question_answer_list]     #For Mistral-7B-Instruct
    # prompt = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a Question-answering assistant, only answer the question.<|eot_id|><|start_header_id|>user<|end_header_id|>\nQuestion: {question_answer[0]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for question_answer in generated_question_answer_list]       #For LLaMA3.1-8B-Instruct
    prompt = [f"<|im_start|>system\nYou are a Question-answering assistant, only answer the question.<|im_end|>\n<|im_start|>user\nQuestion: {question_answer[0]} <|im_end|>\n<|im_start|>assistant\n" for question_answer in generated_question_answer_list]       #For Qwen2.5-7B-Instruct
    tokenized_inputs = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding = "longest", return_attention_mask = True)
    tokenized_inputs = tokenized_inputs.to(device)
    N=tokenized_inputs['input_ids'].shape[1]
    outputs = model.generate(**tokenized_inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens = 100, early_stopping=True, num_beams=5)
    predicted_token_ids = outputs['sequences']
    answers = tokenizer.batch_decode(predicted_token_ids[:, N:], skip_special_tokens=True)
    print(answers)
    return answers, outputs['scores']

def calculate_f1_score(reference_answer, answer):
        # Convert answers and reference_answer to sets of words
    answer_set = set(answer.split())
    reference_set = set(reference_answer.split())

    # Calculate precision, recall, and F1 score
    precision = len(answer_set.intersection(reference_set)) / len(answer_set)
    recall = len(answer_set.intersection(reference_set)) / len(reference_set)

    # Handle the case where precision and recall are both zero
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compare_orig_and_regenerated(generated_questions, orig_answer, reg_answers):
    # not_match = []    #When using F1-scoring, list which stores 1 if not matching [hallucinated] otherwise 0
    pin_point_orig_answers = []
    pin_point_reg_answers = []
    for question, reg_answer in zip(generated_questions, reg_answers):
        pin_orig_answer = qa_model(question = question, context = orig_answer)["answer"]
        pin_reg_answer = qa_model(question = question, context = reg_answer)["answer"]
        # f1_score = calculate_f1_score(pin_orig_answer, pin_reg_answer)    #to be used for F1
        # print(pin_orig_answer, pin_reg_answer, f1_score)                  #to be used for F1
        # print(f1_score)                                                   #to be used for F1
        # not_match.append(int(f1_score < 0.6))                             #to be used for F1
        pin_point_orig_answers.append(pin_orig_answer)
        pin_point_reg_answers.append(pin_reg_answer)
    prompt = build_robust_prompt(pin_point_orig_answers, pin_point_reg_answers)     #to be used for Judge-LLM
    result = call_openai_model(prompt)                                              #to be used for Judge-LLM
    return result, pin_point_orig_answers, pin_point_reg_answers

def find_subset_indices(orig_list, subset):
    len_orig = len(orig_list)
    len_sub = len(subset)
    for i in range(len_orig - len_sub + 1):
        if orig_list[i:i + len_sub] == subset:
            return list(range(i, i + len_sub))
    return []

def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum()

def check_with_probability(reg_answers, pin_point_orig_answers, scores, in_hallu):
    # print(pin_point_orig_answers, len(pin_point_orig_answers))
    not_match = [1]*len(pin_point_orig_answers)
    score = tuple(t.cpu() for t in scores)
    for j, orig_answer, pin_point_answer in zip(range(len(pin_point_orig_answers)), reg_answers, pin_point_orig_answers):
        if in_hallu[j] == 1:
            continue
        precise_answer_indices = find_subset_indices(orig_answer, pin_point_answer)
        precise_answer_tokens_ids = tokenizer(orig_answer[precise_answer_indices[0]: precise_answer_indices[0] + len(precise_answer_indices)])
        precise_answer_tokens = tokenizer.convert_ids_to_tokens(precise_answer_tokens_ids['input_ids'])
        # print(precise_answer_tokens)
        dist = []
        tokenized_words = []
        # print("len of score", len(score))
        for i in range(0, 50):
            try:
                id  = torch.argmax(score[i][j])
                probs = softmax(score[i][j].numpy())
                probs_top = softmax(np.partition(score[i][j], -5)[-5:])
                # ks_statistic, ks_p_value = kstest(probs, 'uniform', args=(probs.min(), probs.max()))
                ks_statistic, ks_p_value = kstest(probs_top, 'uniform', args=(probs_top.min(), probs_top.max()))
                tokenized_words.append(tokenizer.convert_ids_to_tokens(id.item()))
                if ks_p_value > 0.05:
                    dist.append(1)
                    # print(tokenizer.convert_ids_to_tokens(id.item()), 1, "U")
                else:
                    dist.append(0)
                    # print(tokenizer.convert_ids_to_tokens(id.item()), 0, "N-U")
                
            except:
                continue
        if len(precise_answer_tokens) > 1:
        # print(tokenized_words, precise_answer_tokens)
            indices = find_subset_indices(tokenized_words, precise_answer_tokens[1:])
            if len(indices) == 0:
                not_match[j]  = 1
                break
            indices = [indices[0]  - 1] + indices
            # print(indices)
        else: indices = find_subset_indices(tokenized_words, precise_answer_tokens)
        dist_concern = [dist[index] for index in indices]
        if sum(dist_concern) == 0:
            not_match[j] = 0
        else: not_match[j] = 1
    return not_match

if __name__ == "__main__":
    # global_stats = []
    stats = []
    output_metrics = args.metrics_file
    with open(args.sentences, 'rb') as f:
        orig_ques_answers = pickle.load(f)
        for question_answer_pair in tqdm(orig_ques_answers[:10]):
            # print("-"*100)
            try:
                print(question_answer_pair[1])
                generated_question_answer_list = generate_questions_based_on_factual_parts(question_answer_pair[1], method)
                print("Generated question answer list: ")
                print(generated_question_answer_list)
                regenerated_answers, scores = generate_pinpointed_answers(generated_question_answer_list, device)
                print("Regenerated answers")
                print(regenerated_answers)
                generated_questions = [generated_question_answer[0] for generated_question_answer in generated_question_answer_list]
                initial_hallu = compare_orig_and_regenerated(generated_questions, question_answer_pair[1], regenerated_answers)
                print("Initial result (without probability consideration): ", initial_hallu)
                final_hallu = check_with_probability(regenerated_answers, initial_hallu[2], scores, initial_hallu[0])
                print("Final result: ", final_hallu)
                stats.append([generated_questions, initial_hallu, final_hallu]) 
            except:
                print("Exception!")
                stats.append([])
            with open(output_metrics, 'wb') as file:
                pickle.dump(stats, file)
