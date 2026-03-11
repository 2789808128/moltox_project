import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictRequest, PredictResponse
from src.api.model_router import ModelRouter


project_root = r"E:\Project\moltox_project"

app = FastAPI(
    title="MolTox Prediction API",
    description="A unified API for Transformer / Morgan LogReg / Morgan RF / Fusion toxicity prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = ModelRouter(project_root=project_root)


@app.get("/")
def root():
    return {
        "message": "MolTox Prediction API is running",
        "available_models": [
            "transformer",
            "morgan_logreg",
            "morgan_rf",
            "fusion"
        ]
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = router.predict(
            model_type=request.model_type,
            smiles=request.smiles
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")