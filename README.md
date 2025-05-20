# ConFactCheck

This Python script detects hallucinations in model-generated outputs.

## Features

- Extracts factual components using POS tags, NER, or randomly.
- Generates questions targeting those components using a pretrained question-generation model.
- Uses a QA pipeline to extract pinpointed answers from original and regenerated contexts.
- Compares answer pairs using an LLM to judge factual consistency.
- Supports multiple LLMs (e.g., Qwen, LLaMA3, Mistral).
- Includes support for coreference resolution, sentence splitting, and F1-score evaluation.

## Requirements

Install the following Python packages before running:

```bash
pip install torch transformers spacy stanza openai tqdm datasets
python -m spacy download en_core_web_sm
```
## Arguments

python main.py --device "cuda:0" --method "ner" --model_name "Qwen/Qwen1.5-7B-Chat" --model_cached_pth "./cache" --sentences "./hallucinated_sentences.pkl" --metrics_file "./metrics_output.pkl"
    
Argument Descriptions
--device: Target device to run inference (e.g., cuda or cpu).
--method: Method to extract factual parts (ner, pos, random).
--model_name: Name or path of the model used for sentence generation.
--model_cached_pth: Cache directory for loading models.
--sentences: Path to pickle file containing hallucinated sentences.
--metrics_file: File path where evaluation metrics will be stored.



Note: You must provide your OpenAI API key by modifying this line:
```
client = OpenAI(api_key = "<INSERT-API-KEY>")
```
Output:

Pinpointed answers for each extracted fact from both original and regenerated contexts.

Comparison results in the form of a binary list (0 = match, 1 = hallucination).

Evaluation metrics saved to the file specified by --metrics_file.

Notes
This script assumes that the input hallucinated sentences and original references are already aligned and loaded from a pickle file.

Currently optimized for models compatible with Hugging Face's transformers library.
