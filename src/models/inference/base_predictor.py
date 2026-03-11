from abc import ABC, abstractmethod
from pathlib import Path

from src.utils.paths import get_project_root


class BasePredictor(ABC):
    def __init__(self, project_root: str | None = None):
        self.project_root = Path(project_root) if project_root else get_project_root()

    @abstractmethod
    def predict(self, smiles: str) -> dict:
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        pass

    def format_result(self, model_name: str, smiles: str, task_probs: dict, task_preds: dict) -> dict:
        return {
            "model_name": model_name,
            "smiles": smiles,
            "task_probs": task_probs,
            "task_preds": task_preds,
        }