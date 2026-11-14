# Movie Recommendation Engine: Hybrid SVD & Neural Collaborative Filtering

![PyTorch](https://img.shields.io/badge/PyTorch-NCF-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVD-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A personalized multi-stage recommendation service blending matrix factorization with neural collaborative embeddings and content attributes.

## Architecture
```mermaid
flowchart LR
    User[User History] --> SVD[SVD Candidate Generator]
    User --> NCF[Neural Collaborative Filtering]
    SVD & NCF --> Blend[Hybrid Ranker]
    Blend --> TopK[Top-K Movie Cards]
```

## Start API
```bash
uvicorn src.api.service:app --reload
curl http://localhost:8000/recommend/42?count=5
```
