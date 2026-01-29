import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .predictor import load_predictor
from .schema import REQUIRED_COLUMNS, SchemaError, validate_dataframe

logger = logging.getLogger(__name__)

class PredcitRequest(BaseModel):
    records: List[Dict[str,Any]] = Field(..., min_length=1) # en azi 1 meluamt olmali, ... melumat qayidacaq demekdi

class PredictResponse(BaseModel):
    predictions: List[float]
    drift: Optional[Dict[str, Any]] = None
    latency_ms: float

def _load_serve_cfg() -> dict:
    return yaml.safe_load(Path("confgis/serve.yaml").read_text(encoding="utf-8"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    serve_cfg = _load_serve_cfg() #yukle

    model_path = serve_cfg["artifactts"]["model_path"]
    profile_path = serve_cfg["artifactts"].get("training_profile_path")

    pred = load_predictor(model_path = model_path, training_profile_path=profile_path)
    pred.allow_extra_columns = bool(serve_cfg["behavior"].get("allow_extra_columns",False))
    pred.strict_categories = bool(serve_cfg["behavior"].get("strict_categories", False))

    app.state.serve_cfg = serve_cfg
    app.state.predictor = pred

    logger.info("Service started. model=%s profile=%s profile_loaded=%s",
                model_path, profile_path, "yes" if pred.training_profile else "no")

    yield


    logger.info("Service shutting down.")