"""Hybrid recommender combining SVD latent scores and genre content matching."""
import numpy as np

class HybridRecommender:
    def __init__(self, svd_model, alpha: float = 0.7):
        self.svd_model = svd_model
        self.alpha = alpha

    def recommend(self, user_id: int, user_genre_affinity: np.ndarray, top_k: int = 5):
        cf_scores = self.svd_model.predict_user_scores(user_id)
        # Normalize CF scores
        cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-8)
        hybrid_scores = self.alpha * cf_norm + (1 - self.alpha) * user_genre_affinity[:len(cf_norm)]
        return np.argsort(hybrid_scores)[::-1][:top_k].tolist()
