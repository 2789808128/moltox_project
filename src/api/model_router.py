from src.models.inference.predict_transformer import TransformerPredictor
from src.models.inference.predict_logreg import LogRegPredictor
from src.models.inference.predict_rf import RFPredictor
from src.models.inference.predict_fusion import FusionPredictor


class ModelRouter:
    def __init__(self, project_root: str | None = None):
        self.project_root = project_root
        self._predictors = {}

    def _get_predictor(self, model_type: str):
        if model_type not in self._predictors:
            if model_type == "transformer":
                self._predictors[model_type] = TransformerPredictor(project_root=self.project_root)
            elif model_type == "morgan_logreg":
                self._predictors[model_type] = LogRegPredictor(project_root=self.project_root)
            elif model_type == "morgan_rf":
                self._predictors[model_type] = RFPredictor(project_root=self.project_root)
            elif model_type == "fusion":
                self._predictors[model_type] = FusionPredictor(project_root=self.project_root)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

        return self._predictors[model_type]

    def predict(self, model_type: str, smiles: str):
        predictor = self._get_predictor(model_type)
        return predictor.predict(smiles)

    def get_model_metadata(self, model_type: str):
        predictor = self._get_predictor(model_type)
        return predictor.get_metadata()

    def list_models(self):
        return {
            "transformer": "SMILES Transformer",
            "morgan_logreg": "Morgan + Logistic Regression",
            "morgan_rf": "Morgan + Random Forest",
            "fusion": "SMILES + Morgan Fusion",
        }