import os
import pandas as pd
from joblib import load
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from schema.payload import GnenrateRequest,GenerateRespones

models = {}

async def lifespna(app : FastAPI):
    models["model_pipeline"] = load("model.pipeline.pkl")

    yield

    models.clear()

app = FastAPI(lifespan=lifespna)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/predict",response_model=GenerateRespones)
def predict(data : GnenrateRequest):
    user_df = pd.DataFrame([data.dict()])

    pred = models["model_pipeline"].predict(user_df)

    return GenerateRespones(
        pred_total = pred
    )

