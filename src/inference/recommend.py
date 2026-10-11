"""Candidate retrieval and Top-K recommendation pipeline."""
from src.data.loader import generate_recsys_data
from src.models.collaborative_svd import SVDRecommender

class ProductionMovieService:
    def __init__(self):
        ratings, movies = generate_recsys_data()
        self.movies = movies
        self.svd = SVDRecommender(n_components=8)
        self.svd.fit(ratings)

    def get_user_recommendations(self, user_id: int, k: int = 5):
        scores = self.svd.predict_user_scores(user_id)
        top_ids = scores.argsort()[::-1][:k]
        recs = self.movies[self.movies["movie_id"].isin(top_ids)].to_dict(orient="records")
        return recs
