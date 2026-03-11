import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


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


def build_morgan_generator(radius=2, n_bits=2048):
    return GetMorganGenerator(radius=radius, fpSize=n_bits)


def smiles_to_fp(smiles: str, morgan_generator, n_bits: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)

    fp = morgan_generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_matrix(df: pd.DataFrame, radius=2, n_bits=2048):
    smiles_col = "smiles" if "smiles" in df.columns else "SMILES"
    morgan_generator = build_morgan_generator(radius=radius, n_bits=n_bits)
    features = [smiles_to_fp(s, morgan_generator, n_bits) for s in df[smiles_col].tolist()]
    return np.array(features, dtype=np.float32)


def train_logistic_regression_models(train_df: pd.DataFrame, radius=2, n_bits=2048):
    X_train_all = build_feature_matrix(train_df, radius=radius, n_bits=n_bits)
    models = {}

    for task_name in TASK_NAMES:
        label_mask = train_df[task_name].notna().values
        X_train = X_train_all[label_mask]
        y_train = train_df.loc[label_mask, task_name].astype(int).values

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


def train_random_forest_models(train_df: pd.DataFrame, radius=2, n_bits=2048):
    X_train_all = build_feature_matrix(train_df, radius=radius, n_bits=n_bits)
    models = {}

    for task_name in TASK_NAMES:
        label_mask = train_df[task_name].notna().values
        X_train = X_train_all[label_mask]
        y_train = train_df.loc[label_mask, task_name].astype(int).values

        if len(np.unique(y_train)) < 2:
            models[task_name] = None
            continue

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        models[task_name] = model

    return models


def evaluate_models(models: dict, df: pd.DataFrame, radius=2, n_bits=2048):
    X_all = build_feature_matrix(df, radius=radius, n_bits=n_bits)
    task_auc = {}

    for task_name in TASK_NAMES:
        label_mask = df[task_name].notna().values
        X = X_all[label_mask]
        y = df.loc[label_mask, task_name].astype(int).values

        model = models.get(task_name)
        if model is None or len(np.unique(y)) < 2:
            task_auc[task_name] = None
            continue

        probs = model.predict_proba(X)[:, 1]
        task_auc[task_name] = roc_auc_score(y, probs)

    valid_aucs = [v for v in task_auc.values() if v is not None]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else None

    return task_auc, mean_auc