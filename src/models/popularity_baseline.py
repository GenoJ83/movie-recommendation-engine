"""Non-personalized popularity and Bayes average baseline."""
import pandas as pd
from src.data.loader import generate_recsys_data

def get_popular_movies(top_n: int = 10):
    ratings, movies = generate_recsys_data()
    agg = ratings.groupby("movie_id").agg(count=("rating", "count"), mean_rating=("rating", "mean"))
    # Minimum 10 ratings for recommendation confidence
    popular = agg[agg["count"] >= 10].sort_values(by="mean_rating", ascending=False).head(top_n)
    return popular.index.tolist()

if __name__ == "__main__":
    top = get_popular_movies(5)
    print("Top Popular Movies:", top)
