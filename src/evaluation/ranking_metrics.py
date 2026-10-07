"""Ranking evaluation metrics: Precision@K, Recall@K, NDCG@K."""
import numpy as np

def compute_precision_recall_at_k(recommended, ground_truth, k: int = 10):
    rec_k = set(recommended[:k])
    gt = set(ground_truth)
    if not gt:
        return 0.0, 0.0
    hits = len(rec_k & gt)
    precision = hits / k
    recall = hits / len(gt)
    return round(precision, 4), round(recall, 4)

def compute_ndcg_at_k(recommended, ground_truth, k: int = 10):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    return round(dcg / idcg if idcg > 0 else 0.0, 4)
