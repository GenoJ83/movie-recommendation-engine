"""Unit tests for matrices, SVD model, and ranking metrics."""
from src.data.loader import generate_recsys_data
from src.models.collaborative_svd import SVDRecommender
from src.evaluation.ranking_metrics import compute_precision_recall_at_k, compute_ndcg_at_k

def test_data_loader():
    r, m = generate_recsys_data(100, 50, 1000)
    assert len(r) > 0
    assert len(m) == 50

def test_svd_scoring():
    r, _ = generate_recsys_data(50, 30, 500)
    svd = SVDRecommender(n_components=4)
    svd.fit(r)
    scores = svd.predict_user_scores(0)
    assert len(scores) == 30

def test_ranking_metrics():
    rec = [1, 2, 3, 4, 5]
    gt = [2, 5, 8]
    p, r = compute_precision_recall_at_k(rec, gt, k=5)
    assert p == 0.4
    ndcg = compute_ndcg_at_k(rec, gt, k=5)
    assert ndcg > 0.0
