"""Construct sparse user-item interaction matrices and genre one-hot encodings."""
import pandas as pd
from scipy.sparse import csr_matrix

def build_interaction_matrix(ratings_df: pd.DataFrame, n_users: int, n_items: int):
    rows = ratings_df["user_id"].values
    cols = ratings_df["movie_id"].values
    vals = ratings_df["rating"].values
    matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    return matrix
