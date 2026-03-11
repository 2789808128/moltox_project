import joblib
import numpy as np

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.models.inference.base_predictor import BasePredictor


class LogRegPredictor(BasePredictor):
    def __init__(self, project_root: str | None = None):
        super().__init__(project_root=project_root)

        self.model_path = self.project_root / "outputs" / "checkpoints" / "ml_baselines" / "morgan_logreg.joblib"
        saved = joblib.load(self.model_path)

        self.models = saved["models"]
        self.task_names = saved["task_names"]
        self.fp_radius = saved["fp_radius"]
        self.fp_n_bits = saved["fp_n_bits"]
        self.threshold = saved["threshold"]
        self.model_name = saved["model_name"]

        self.morgan_generator = GetMorganGenerator(
            radius=self.fp_radius,
            fpSize=self.fp_n_bits
        )

    def _smiles_to_fp(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        fp = self.morgan_generator.GetFingerprint(mol)
        arr = np.zeros((self.fp_n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def predict(self, smiles: str):
        x = self._smiles_to_fp(smiles).reshape(1, -1)

        task_probs = {}
        task_preds = {}

        for task_name in self.task_names:
            model = self.models.get(task_name)
            if model is None:
                prob = None
                pred = None
            else:
                prob = float(model.predict_proba(x)[0, 1])
                pred = 1 if prob >= self.threshold else 0

            task_probs[task_name] = prob
            task_preds[task_name] = pred

        return self.format_result(
            model_name=self.model_name,
            smiles=smiles,
            task_probs=task_probs,
            task_preds=task_preds,
        )

    def get_metadata(self):
        return {
            "model_type": "morgan_logreg",
            "model_name": self.model_name,
            "task_names": self.task_names,
            "threshold": self.threshold,
            "input_type": "morgan_fingerprint",
        }


if __name__ == "__main__":
    predictor = LogRegPredictor()
    result = predictor.predict("CCO")
    print(result)