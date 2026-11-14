"""Collaborative filtering using Truncated SVD matrix factorization."""
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from src.data.loader import generate_recsys_data
from src.features.matrices import build_interaction_matrix

class SVDRecommender:
    def __init__(self, n_components: int = 16):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

    def fit(self, ratings_df: pd.DataFrame):
        n_users = ratings_df["user_id"].max() + 1
        n_items = ratings_df["movie_id"].max() + 1
        matrix = build_interaction_matrix(ratings_df, n_users, n_items)
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_
        return self

    def predict_user_scores(self, user_id: int):
        u_emb = self.user_factors[user_id]
        scores = np.dot(u_emb, self.item_factors)
        return scores
