import argparse
import pickle
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    confusion_matrix, auc, average_precision_score,
    precision_recall_curve, roc_auc_score
)
from scipy.stats import pearsonr
from tqdm import tqdm


def calculate_f1_score(reference_answer, answer):
    """Compute F1 score between two text answers."""
    answer_set = set(answer.lower().split())
    reference_set = set(reference_answer.lower().split())

    if not answer_set or not reference_set:
        return 0.0

    intersection = answer_set.intersection(reference_set)
    precision = len(intersection) / len(answer_set)
    recall = len(intersection) / len(reference_set)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_model(golden_hallucination, model_outputs):
    """Evaluate hallucination detection using standard metrics."""
    binary_preds = []
    confidence_scores = []

    for output in model_outputs:
        if len(output) == 0:
            binary_preds.append(1)  # Consider empty as hallucinated
            confidence_scores.append(1.0)
        else:
            score = sum(output) / len(output)
            binary_preds.append(0 if score < 1.0 else 1)
            confidence_scores.append(score)

    # Confusion matrix
    confusion = confusion_matrix(golden_hallucination, binary_preds)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Metrics
    precision, recall, _ = precision_recall_curve(golden_hallucination, confidence_scores)
    auc_pr = auc(recall, precision)
    auc_roc = roc_auc_score(golden_hallucination, confidence_scores)
    avg_precision = average_precision_score(golden_hallucination, confidence_scores)
    pearson_corr, _ = pearsonr(golden_hallucination, confidence_scores)

    # Print results
    print(f"TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Avg. Precision: {avg_precision:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")


def main(args):
    # Load golden answers (not used in evaluation here, but retained for reference)
    dataset = load_dataset("google-research-datasets/nq_open")
    golden_answer_webq = dataset["validation"]["answer"]

    # Load ground truth hallucination labels
    with open(args.golden_hallucination_file, 'rb') as f:
        golden_hallucination = pickle.load(f)

    # Load model results
    with open(args.model_outputs_file, 'rb') as f:
        model_outputs = pickle.load(f)

    # Evaluate
    evaluate_model(golden_hallucination, model_outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hallucination detection model outputs.")
    parser.add_argument(
        "--golden_hallucination_file",
        type=str,
        required=True,
        help="Path to the pickle file containing ground truth hallucination labels"
    )
    parser.add_argument(
        "--model_outputs_file",
        type=str,
        required=True,
        help="Path to the pickle file containing model output scores"
    )
    args = parser.parse_args()
    main(args)
