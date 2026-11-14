"""Synthetic user-item rating matrix and movie metadata generator."""
import pandas as pd
import numpy as np

GENRES = ["Action", "Sci-Fi", "Drama", "Comedy", "Thriller", "Horror"]

def generate_recsys_data(n_users: int = 500, n_items: int = 100, n_ratings: int = 5000):
    rng = np.random.default_rng(42)
    users = rng.integers(0, n_users, size=n_ratings)
    items = rng.integers(0, n_items, size=n_ratings)
    ratings = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n_ratings, p=[0.1, 0.1, 0.25, 0.35, 0.2])

    ratings_df = pd.DataFrame({"user_id": users, "movie_id": items, "rating": ratings}).drop_duplicates(subset=["user_id", "movie_id"])
    
    movies_df = pd.DataFrame({
        "movie_id": list(range(n_items)),
        "title": [f"Movie Title {i}" for i in range(n_items)],
        "genre": [rng.choice(GENRES) for _ in range(n_items)]
    })
    return ratings_df, movies_df

if __name__ == "__main__":
    r, m = generate_recsys_data()
    print(f"Loaded {len(r)} ratings across {len(m)} catalog movies.")
