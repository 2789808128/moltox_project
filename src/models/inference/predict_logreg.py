import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.linear_model import LogisticRegression


TASK_NAMES = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


class LogRegPredictor:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.train_csv = os.path.join(project_root, "data", "processed", "tox21_train.csv")

        self.fp_radius = 2
        self.fp_n_bits = 2048
        self.threshold = 0.5

        self.morgan_generator = GetMorganGenerator(
            radius=self.fp_radius,
            fpSize=self.fp_n_bits
        )

        self.train_df = pd.read_csv(self.train_csv)
        self.X_train_all = self._build_feature_matrix(self.train_df)

        # 每个任务一个模型
        self.models = self._train_models()

    def _smiles_to_fp(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((self.fp_n_bits,), dtype=np.float32)

        fp = self.morgan_generator.GetFingerprint(mol)
        arr = np.zeros((self.fp_n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _build_feature_matrix(self, df: pd.DataFrame):
        features = []
        for smiles in df["smiles"].tolist():
            features.append(self._smiles_to_fp(smiles))
        return np.array(features, dtype=np.float32)

    def _train_models(self):
        models = {}

        for task_name in TASK_NAMES:
            label_mask = self.train_df[task_name].notna().values
            X_train = self.X_train_all[label_mask]
            y_train = self.train_df.loc[label_mask, task_name].astype(int).values

            if len(np.unique(y_train)) < 2:
                models[task_name] = None
                continue

            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42
            )
            model.fit(X_train, y_train)
            models[task_name] = model

        return models

    def predict(self, smiles: str):
        x = self._smiles_to_fp(smiles).reshape(1, -1)

        task_probs = {}
        task_preds = {}

        for task_name in TASK_NAMES:
            model = self.models[task_name]
            if model is None:
                prob = None
                pred = None
            else:
                prob = float(model.predict_proba(x)[0, 1])
                pred = 1 if prob >= self.threshold else 0

            task_probs[task_name] = prob
            task_preds[task_name] = pred

        return {
            "model_name": "morgan_logreg",
            "smiles": smiles,
            "task_probs": task_probs,
            "task_preds": task_preds,
        }


if __name__ == "__main__":
    project_root = r"E:\Project\moltox_project"

    predictor = LogRegPredictor(project_root=project_root)

    smiles = "CCO"
    result = predictor.predict(smiles)

    print("Model:", result["model_name"])
    print("SMILES:", result["smiles"])
    print("\nTask probabilities:")
    for k, v in result["task_probs"].items():
        print(f"{k}: {v}")

    print("\nTask predictions:")
    for k, v in result["task_preds"].items():
        print(f"{k}: {v}")