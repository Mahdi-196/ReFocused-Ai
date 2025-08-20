#!/usr/bin/env python3
"""
Metrics for fine-tuning evaluation
"""

import numpy as np
from typing import List, Dict, Union, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import torch
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def compute_fine_tuning_metrics(
    predictions: Union[List, np.ndarray],
    labels: Union[List, np.ndarray],
    task_type: str = "custom"
) -> Dict[str, float]:
    """Compute metrics based on task type"""
    
    # Convert to numpy arrays if needed
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(labels, list):
        labels = np.array(labels)
    
    # Flatten if needed
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if labels.ndim > 1:
        labels = labels.flatten()
    
    # Filter out padding tokens (-100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    if len(predictions) == 0:
        return {"accuracy": 0.0, "perplexity": float('inf')}
    
    # Base metrics
    metrics = {}
    
    if task_type == "chat":
        metrics.update(compute_chat_metrics(predictions, labels))
    elif task_type == "code":
        metrics.update(compute_code_metrics(predictions, labels))
    elif task_type == "instruct":
        metrics.update(compute_instruction_metrics(predictions, labels))
    else:
        metrics.update(compute_general_metrics(predictions, labels))
    
    return metrics


def compute_general_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute general language modeling metrics"""
    
    # Token-level accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Compute perplexity approximation
    # This is a simplified version - for true perplexity, we'd need the loss
    unique_predictions = len(np.unique(predictions))
    vocab_usage = unique_predictions / max(len(predictions), 1)
    
    metrics = {
        "accuracy": float(accuracy),
        "vocab_usage": float(vocab_usage),
        "unique_tokens": int(unique_predictions),
        "total_tokens": len(predictions)
    }
    
    return metrics


def compute_chat_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute metrics specific to chat fine-tuning"""
    
    base_metrics = compute_general_metrics(predictions, labels)
    
    # Additional chat-specific metrics
    # Response diversity (simplified - ratio of unique tokens)
    response_diversity = len(np.unique(predictions)) / len(predictions)
    
    # Turn coherence (simplified - measure of consistent token patterns)
    # Look for repeated patterns that might indicate proper turn-taking
    bigrams = list(zip(predictions[:-1], predictions[1:]))
    unique_bigrams = len(set(bigrams))
    bigram_diversity = unique_bigrams / max(len(bigrams), 1)
    
    chat_metrics = {
        **base_metrics,
        "response_diversity": float(response_diversity),
        "bigram_diversity": float(bigram_diversity),
    }
    
    return chat_metrics


def compute_code_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute metrics specific to code fine-tuning"""
    
    base_metrics = compute_general_metrics(predictions, labels)
    
    # Code-specific metrics
    # Exact match rate for common code tokens (simplified)
    # In practice, you'd want to identify actual code tokens
    exact_matches = np.sum(predictions == labels)
    exact_match_rate = exact_matches / len(labels)
    
    # Syntax pattern consistency (simplified)
    # Look for common code patterns (brackets, parentheses, etc.)
    # This is a placeholder - real implementation would use tokenizer
    pattern_consistency = compute_pattern_consistency(predictions, labels)
    
    code_metrics = {
        **base_metrics,
        "exact_match_rate": float(exact_match_rate),
        "pattern_consistency": float(pattern_consistency),
    }
    
    return code_metrics


def compute_instruction_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute metrics specific to instruction fine-tuning"""
    
    base_metrics = compute_general_metrics(predictions, labels)
    
    # Instruction-following metrics
    # Response completeness (simplified - based on length)
    avg_response_length = len(predictions) / max(len(np.unique(labels)), 1)
    
    # Instruction adherence (simplified - similarity to expected patterns)
    adherence_score = compute_sequence_similarity(predictions, labels)
    
    instruction_metrics = {
        **base_metrics,
        "avg_response_length": float(avg_response_length),
        "adherence_score": float(adherence_score),
    }
    
    return instruction_metrics


def compute_pattern_consistency(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute pattern consistency score"""
    
    # Create n-grams
    n = 3
    pred_ngrams = [tuple(predictions[i:i+n]) for i in range(len(predictions)-n+1)]
    label_ngrams = [tuple(labels[i:i+n]) for i in range(len(labels)-n+1)]
    
    # Count common n-grams
    pred_counter = Counter(pred_ngrams)
    label_counter = Counter(label_ngrams)
    
    # Compute overlap
    common_ngrams = set(pred_counter.keys()) & set(label_counter.keys())
    if len(label_counter) == 0:
        return 0.0
    
    consistency = len(common_ngrams) / len(label_counter)
    return consistency


def compute_sequence_similarity(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute sequence similarity score"""
    
    # Simple sequence alignment score
    matches = np.sum(predictions == labels)
    total = len(labels)
    
    similarity = matches / total if total > 0 else 0.0
    return similarity


def compute_generation_metrics(
    generated_texts: List[str],
    reference_texts: List[str],
    tokenizer=None
) -> Dict[str, float]:
    """Compute metrics for text generation quality"""
    
    metrics = {}
    
    # Length statistics
    gen_lengths = [len(text.split()) for text in generated_texts]
    ref_lengths = [len(text.split()) for text in reference_texts]
    
    metrics["avg_gen_length"] = np.mean(gen_lengths)
    metrics["avg_ref_length"] = np.mean(ref_lengths)
    metrics["length_ratio"] = metrics["avg_gen_length"] / max(metrics["avg_ref_length"], 1)
    
    # Diversity metrics
    all_gen_tokens = " ".join(generated_texts).split()
    unique_tokens = len(set(all_gen_tokens))
    total_tokens = len(all_gen_tokens)
    
    metrics["token_diversity"] = unique_tokens / max(total_tokens, 1)
    metrics["unique_tokens"] = unique_tokens
    
    # Repetition metrics
    metrics["repetition_rate"] = compute_repetition_rate(generated_texts)
    
    # BLEU-like score (simplified)
    if reference_texts:
        metrics["bleu_1"] = compute_simple_bleu(generated_texts, reference_texts, n=1)
        metrics["bleu_2"] = compute_simple_bleu(generated_texts, reference_texts, n=2)
    
    return metrics


def compute_repetition_rate(texts: List[str]) -> float:
    """Compute rate of repeated n-grams in generated texts"""
    
    all_trigrams = []
    for text in texts:
        words = text.split()
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        all_trigrams.extend(trigrams)
    
    if not all_trigrams:
        return 0.0
    
    trigram_counts = Counter(all_trigrams)
    repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
    
    repetition_rate = repeated_trigrams / len(trigram_counts)
    return repetition_rate


def compute_simple_bleu(generated: List[str], references: List[str], n: int = 1) -> float:
    """Compute simplified BLEU score"""
    
    scores = []
    
    for gen, ref in zip(generated, references):
        gen_tokens = gen.split()
        ref_tokens = ref.split()
        
        if len(gen_tokens) < n or len(ref_tokens) < n:
            scores.append(0.0)
            continue
        
        # Get n-grams
        gen_ngrams = [tuple(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
        
        # Count matches
        ref_ngram_set = set(ref_ngrams)
        matches = sum(1 for ngram in gen_ngrams if ngram in ref_ngram_set)
        
        # Precision
        precision = matches / len(gen_ngrams) if gen_ngrams else 0.0
        
        # Brevity penalty
        bp = min(1.0, len(gen_tokens) / len(ref_tokens)) if ref_tokens else 0.0
        
        scores.append(precision * bp)
    
    return np.mean(scores) if scores else 0.0


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss"""
    try:
        return float(np.exp(loss))
    except OverflowError:
        return float('inf')


def aggregate_metrics(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """Aggregate metrics across multiple batches"""
    
    if not metrics_list:
        return {}
    
    aggregated = {}
    
    # Get all metric names
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    # Aggregate each metric
    for key in all_keys:
        values = [m.get(key, 0.0) for m in metrics_list if key in m]
        if values:
            aggregated[key] = np.mean(values)
    
    return aggregated


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics for logging"""
    
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("📊 Fine-tuning Metrics Example")
    
    # Simulate some predictions and labels
    predictions = np.random.randint(0, 1000, size=1000)
    labels = np.random.randint(0, 1000, size=1000)
    
    # Compute metrics for different task types
    for task_type in ["general", "chat", "code", "instruct"]:
        print(f"\n🎯 Task: {task_type}")
        metrics = compute_fine_tuning_metrics(predictions, labels, task_type)
        print(format_metrics(metrics, "  ")) 