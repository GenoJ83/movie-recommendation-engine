"""FastAPI recommendation web service."""
from fastapi import FastAPI
from src.inference.recommend import ProductionMovieService

app = FastAPI(title="Movie Recommendation Engine API", version="1.0.0")
service = None

@app.on_event("startup")
def startup():
    global service
    service = ProductionMovieService()

@app.get("/health")
def health():
    return {"status": "operational", "engine": "Hybrid-RecSys"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, count: int = 5):
    return {"user_id": user_id, "recommendations": service.get_user_recommendations(user_id, count)}
