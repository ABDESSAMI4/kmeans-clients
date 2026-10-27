"""FastAPI inference service for the serialized RFM K=3 pipeline."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from customer_segmentation.pipeline import (  # noqa: E402
    META_PATH,
    MODEL_PATH,
    predict_with_recommendation,
)

app = FastAPI(
    title="Customer Segmentation RFM API",
    version="2.0.0",
    description="Affectation d'un client au pipeline RFM K=3 et recommandation descriptive.",
)


class ClientPayload(BaseModel):
    """Raw input schema. Campaign outcomes are accepted as extra JSON keys but ignored."""

    model_config = ConfigDict(extra="ignore")

    Income: float | None = Field(default=None, ge=0)
    Recency: float = Field(ge=0)
    NumDealsPurchases: float = Field(ge=0)
    NumWebPurchases: float = Field(ge=0)
    NumCatalogPurchases: float = Field(ge=0)
    NumStorePurchases: float = Field(ge=0)
    MntWines: float = Field(ge=0)
    MntFruits: float = Field(ge=0)
    MntMeatProducts: float = Field(ge=0)
    MntFishProducts: float = Field(ge=0)
    MntSweetProducts: float = Field(ge=0)
    MntGoldProds: float = Field(ge=0)
    Dt_Customer: str = Field(description="Date d'inscription, format du jeu ou ISO-8601")


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Artefacts RFM absents. Entraîner le modèle avec python -m customer_segmentation.pipeline."
        )
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    try:
        load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "model": "RFM KMeans v2"}


@app.post("/predict")
def predict(payload: ClientPayload) -> dict[str, Any]:
    """Return a cluster and a business action hypothesis for one raw client."""
    try:
        pipeline, metadata = load_artifacts()
        import pandas as pd

        prediction = predict_with_recommendation(
            pd.DataFrame([payload.model_dump()]), pipeline=pipeline, metadata=metadata
        )[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        **prediction,
        "interpretation": "Recommandation descriptive à valider par expérimentation; aucune causalité n'est estimée.",
    }
