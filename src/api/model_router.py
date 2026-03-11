from src.models.inference.predict_transformer import TransformerPredictor
from src.models.inference.predict_logreg import LogRegPredictor
from src.models.inference.predict_rf import RFPredictor
from src.models.inference.predict_fusion import FusionPredictor


class ModelRouter:
    def __init__(self, project_root: str):
        self.project_root = project_root

        # 延迟初始化，避免一启动就把所有模型都加载/训练一遍
        self._predictors = {}

    def _get_predictor(self, model_type: str):
        model_type = model_type.lower().strip()

        if model_type not in self._predictors:
            if model_type == "transformer":
                self._predictors[model_type] = TransformerPredictor(self.project_root)
            elif model_type == "morgan_logreg":
                self._predictors[model_type] = LogRegPredictor(self.project_root)
            elif model_type == "morgan_rf":
                self._predictors[model_type] = RFPredictor(self.project_root)
            elif model_type == "fusion":
                self._predictors[model_type] = FusionPredictor(self.project_root)
            else:
                raise ValueError(
                    "Unsupported model_type. Choose from: "
                    "transformer / morgan_logreg / morgan_rf / fusion"
                )

        return self._predictors[model_type]

    def predict(self, model_type: str, smiles: str):
        predictor = self._get_predictor(model_type)
        return predictor.predict(smiles)


if __name__ == "__main__":
    project_root = r"E:\Project\moltox_project"

    router = ModelRouter(project_root=project_root)

    result = router.predict(model_type="fusion", smiles="CCO")

    print("Model:", result["model_name"])
    print("SMILES:", result["smiles"])
    print("\nTask probabilities:")
    for k, v in result["task_probs"].items():
        print(f"{k}: {v}")