"""Maximal Marginal Relevance (MMR) recommendation reranker."""
import numpy as np

def mmr_rerank(candidate_ids, relevance_scores, item_similarity_matrix, lambda_param: float = 0.6, top_k: int = 5):
    """Greedy MMR selection balancing relevance and novelty."""
    selected = []
    unselected = list(range(len(candidate_ids)))

    while len(selected) < min(top_k, len(candidate_ids)):
        best_score = -float('inf')
        best_idx = None
        for i in unselected:
            relevance = relevance_scores[i]
            if not selected:
                diversity = 0.0
            else:
                diversity = max(item_similarity_matrix[i, j] for j in selected)
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(best_idx)
        unselected.remove(best_idx)

    return [candidate_ids[i] for i in selected]
