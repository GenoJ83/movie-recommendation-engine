"""Interaction matrix sparsity and rating distribution diagnostics."""
from src.data.loader import generate_recsys_data

def analyze_matrix_sparsity():
    ratings, movies = generate_recsys_data()
    n_users = ratings["user_id"].nunique()
    n_items = ratings["movie_id"].nunique()
    total_possible = n_users * n_items
    sparsity = 1.0 - (len(ratings) / total_possible)

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "total_ratings": len(ratings),
        "matrix_sparsity_pct": round(sparsity * 100, 2),
        "mean_rating": round(float(ratings["rating"].mean()), 2)
    }
    print("Matrix Sparsity Report:", stats)
    return stats

if __name__ == "__main__":
    analyze_matrix_sparsity()
