import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

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


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    将单条 SMILES 转成 Morgan fingerprint numpy 向量
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_matrix(df: pd.DataFrame, smiles_col: str = "smiles", radius: int = 2, n_bits: int = 2048):
    """
    对整个 dataframe 一次性生成 Morgan fingerprint 特征矩阵
    返回:
        X: [num_samples, n_bits]
        valid_mask: [num_samples]，表示该行是否成功转成指纹
    """
    features = []
    valid_mask = []

    for smiles in df[smiles_col].tolist():
        fp = smiles_to_morgan_fp(smiles, radius=radius, n_bits=n_bits)
        if fp is None:
            features.append(np.zeros((n_bits,), dtype=np.float32))
            valid_mask.append(False)
        else:
            features.append(fp)
            valid_mask.append(True)

    X = np.array(features, dtype=np.float32)
    valid_mask = np.array(valid_mask, dtype=bool)

    return X, valid_mask


def evaluate_model_auc(model, X_test, y_test):
    """
    计算测试集 AUC
    """
    unique_classes = np.unique(y_test)
    if len(unique_classes) < 2:
        return None

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    return auc


def run_logistic_regression_baseline(train_df, test_df, X_train_all, X_test_all, train_fp_mask, test_fp_mask):
    """
    Morgan FP + Logistic Regression
    """
    results = {}

    print("\n" + "=" * 60)
    print("Running Morgan FP + Logistic Regression")
    print("=" * 60)

    valid_aucs = []

    for task_name in TASK_NAMES:
        print(f"\nTask: {task_name}")

        train_label_mask = train_df[task_name].notna().values
        test_label_mask = test_df[task_name].notna().values

        train_mask = train_label_mask & train_fp_mask
        test_mask = test_label_mask & test_fp_mask

        X_train = X_train_all[train_mask]
        y_train = train_df.loc[train_mask, task_name].astype(int).values

        X_test = X_test_all[test_mask]
        y_test = test_df.loc[test_mask, task_name].astype(int).values

        print(f"  Train samples: {len(y_train)}")
        print(f"  Test samples: {len(y_test)}")

        if len(y_train) == 0 or len(y_test) == 0:
            print("  Skip: no valid samples.")
            results[task_name] = None
            continue

        if len(np.unique(y_train)) < 2:
            print("  Skip: training labels have only one class.")
            results[task_name] = None
            continue

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        )

        model.fit(X_train, y_train)
        auc = evaluate_model_auc(model, X_test, y_test)

        results[task_name] = auc
        print(f"  Test ROC-AUC: {auc}")

        if auc is not None:
            valid_aucs.append(auc)

    mean_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else None
    print("\nLogistic Regression Mean ROC-AUC:", mean_auc)

    return results, mean_auc


def run_random_forest_baseline(train_df, test_df, X_train_all, X_test_all, train_fp_mask, test_fp_mask):
    """
    Morgan FP + Random Forest
    """
    results = {}

    print("\n" + "=" * 60)
    print("Running Morgan FP + Random Forest")
    print("=" * 60)

    valid_aucs = []

    for task_name in TASK_NAMES:
        print(f"\nTask: {task_name}")

        train_label_mask = train_df[task_name].notna().values
        test_label_mask = test_df[task_name].notna().values

        train_mask = train_label_mask & train_fp_mask
        test_mask = test_label_mask & test_fp_mask

        X_train = X_train_all[train_mask]
        y_train = train_df.loc[train_mask, task_name].astype(int).values

        X_test = X_test_all[test_mask]
        y_test = test_df.loc[test_mask, task_name].astype(int).values

        print(f"  Train samples: {len(y_train)}")
        print(f"  Test samples: {len(y_test)}")

        if len(y_train) == 0 or len(y_test) == 0:
            print("  Skip: no valid samples.")
            results[task_name] = None
            continue

        if len(np.unique(y_train)) < 2:
            print("  Skip: training labels have only one class.")
            results[task_name] = None
            continue

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)
        auc = evaluate_model_auc(model, X_test, y_test)

        results[task_name] = auc
        print(f"  Test ROC-AUC: {auc}")

        if auc is not None:
            valid_aucs.append(auc)

    mean_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else None
    print("\nRandom Forest Mean ROC-AUC:", mean_auc)

    return results, mean_auc


def main():
    project_root = r"E:\Project\moltox_project"

    train_csv = os.path.join(project_root, "data", "processed", "tox21_train.csv")
    test_csv = os.path.join(project_root, "data", "processed", "tox21_test.csv")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nBuilding Morgan fingerprint features once for train/test ...")
    X_train_all, train_fp_mask = build_feature_matrix(train_df, radius=2, n_bits=2048)
    X_test_all, test_fp_mask = build_feature_matrix(test_df, radius=2, n_bits=2048)

    print("Train feature matrix shape:", X_train_all.shape)
    print("Test feature matrix shape:", X_test_all.shape)
    print("Valid train fingerprints:", int(train_fp_mask.sum()))
    print("Valid test fingerprints:", int(test_fp_mask.sum()))

    # 1) Logistic Regression
    lr_results, lr_mean_auc = run_logistic_regression_baseline(
        train_df, test_df, X_train_all, X_test_all, train_fp_mask, test_fp_mask
    )

    # 2) Random Forest
    rf_results, rf_mean_auc = run_random_forest_baseline(
        train_df, test_df, X_train_all, X_test_all, train_fp_mask, test_fp_mask
    )

    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    print("Transformer Test Mean ROC-AUC: 0.7554")
    print(f"Logistic Regression Mean ROC-AUC: {lr_mean_auc}")
    print(f"Random Forest Mean ROC-AUC: {rf_mean_auc}")


if __name__ == "__main__":
    main()