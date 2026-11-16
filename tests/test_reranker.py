"""Tests for MMR reranker."""
import numpy as np
from src.models.mmr_reranker import mmr_rerank

def test_mmr_reranking():
    candidates = [101, 102, 103, 104]
    relevance = [0.95, 0.90, 0.85, 0.80]
    sim = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.2],
        [0.1, 0.1, 0.2, 1.0]
    ])
    recs = mmr_rerank(candidates, relevance, sim, lambda_param=0.5, top_k=2)
    assert len(recs) == 2
    assert recs[0] == 101
    assert recs[1] == 103 # Diversified choice over 102
