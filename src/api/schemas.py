from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    model_type: str = Field(..., description="模型类型: transformer / morgan_logreg / morgan_rf / fusion")
    smiles: str = Field(..., description="输入的 SMILES 字符串")


class PredictResponse(BaseModel):
    model_name: str
    smiles: str
    task_probs: Dict[str, Optional[float]]
    task_preds: Dict[str, Optional[int]]