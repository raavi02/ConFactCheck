import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pickle
import os


def load_model_and_tokenizer(model_name: str, cache_dir: str, device: str):
    """Loads the LLaMA tokenizer and model with appropriate settings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model = model.to(device)

    return tokenizer, model


def generate_answers(tokenizer, model, questions, device: str, output_path: str):
    """Generates answers for each question and stores results to file."""
    results = []
    for question in tqdm(questions, desc="Generating answers"):
        # Build prompt
        prompt = [
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a Question-answering assistant, only answer the question."
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ]

        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt", padding="longest").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=100
            )
        predicted_token_ids = outputs["sequences"]
        answer = tokenizer.batch_decode(predicted_token_ids[:, prompt_len:], skip_special_tokens=True)[0]

        results.append((question, answer))

        # Save progress
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

    print(f"Finished generation. Saved {len(results)} results to {output_path}")


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer, model = load_model_and_tokenizer(args.model_name, args.cache_dir, device)

    # Load dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    questions = dataset[args.split]['question']

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Generate and save answers
    generate_answers(tokenizer, model, questions, device, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers using LLaMA-3 model.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model identifier from HuggingFace hub.")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Path to the cache directory for model weights.")
    parser.add_argument("--dataset_name", type=str, default="hotpot_qa",
                        help="Dataset name from HuggingFace Datasets.")
    parser.add_argument("--dataset_config", type=str, default="fullwiki",
                        help="Configuration of the dataset.")
    parser.add_argument("--split", type=str, default="validation",
                        help="Dataset split to use (e.g., train, validation, test).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the generated question-answer pairs (as a .pkl file).")

    args = parser.parse_args()
    main(args)
