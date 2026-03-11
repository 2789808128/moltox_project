import os
import joblib
import pandas as pd

from src.models.ml_baseline import (
    TASK_NAMES,
    train_logistic_regression_models,
    train_random_forest_models,
    evaluate_models,
)
from src.utils.experiment_logger import append_experiment_record


def main():
    project_root = r"E:\Project\moltox_project"

    train_csv = os.path.join(project_root, "data", "processed", "tox21_train.csv")
    test_csv = os.path.join(project_root, "data", "processed", "tox21_test.csv")

    model_dir = os.path.join(project_root, "outputs", "checkpoints", "ml_baselines")
    os.makedirs(model_dir, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    fp_radius = 2
    fp_n_bits = 2048

    # -------------------------
    # Logistic Regression
    # -------------------------
    print("Training Logistic Regression models...")
    logreg_models = train_logistic_regression_models(
        train_df=train_df,
        radius=fp_radius,
        n_bits=fp_n_bits
    )

    logreg_task_auc, logreg_mean_auc = evaluate_models(
        models=logreg_models,
        df=test_df,
        radius=fp_radius,
        n_bits=fp_n_bits
    )

    logreg_path = os.path.join(model_dir, "morgan_logreg.joblib")
    joblib.dump(
        {
            "models": logreg_models,
            "task_names": TASK_NAMES,
            "fp_radius": fp_radius,
            "fp_n_bits": fp_n_bits,
            "threshold": 0.5,
            "model_name": "morgan_logreg",
            "test_task_auc": logreg_task_auc,
            "test_mean_auc": logreg_mean_auc,
        },
        logreg_path
    )
    print("Saved:", logreg_path)
    print(f"LogReg Test Mean ROC-AUC: {logreg_mean_auc:.4f}")

    exp_id = append_experiment_record(
        project_root=project_root,
        model_type="morgan_logreg",
        stage="test",
        train_script="src/models/train_ml_models.py",
        test_script="src/models/inference/predict_logreg.py",
        best_epoch="",
        best_valid_auc=logreg_mean_auc,
        test_mean_auc=logreg_mean_auc,
        learning_rate="",
        batch_size="",
        num_epochs="",
        notes="auto-logged from train_ml_models.py after offline training and evaluation",
    )
    print(f"Experiment record saved: {exp_id}")

    # -------------------------
    # Random Forest
    # -------------------------
    print("\nTraining Random Forest models...")
    rf_models = train_random_forest_models(
        train_df=train_df,
        radius=fp_radius,
        n_bits=fp_n_bits
    )

    rf_task_auc, rf_mean_auc = evaluate_models(
        models=rf_models,
        df=test_df,
        radius=fp_radius,
        n_bits=fp_n_bits
    )

    rf_path = os.path.join(model_dir, "morgan_rf.joblib")
    joblib.dump(
        {
            "models": rf_models,
            "task_names": TASK_NAMES,
            "fp_radius": fp_radius,
            "fp_n_bits": fp_n_bits,
            "threshold": 0.5,
            "model_name": "morgan_rf",
            "test_task_auc": rf_task_auc,
            "test_mean_auc": rf_mean_auc,
        },
        rf_path
    )
    print("Saved:", rf_path)
    print(f"RF Test Mean ROC-AUC: {rf_mean_auc:.4f}")

    exp_id = append_experiment_record(
        project_root=project_root,
        model_type="morgan_rf",
        stage="test",
        train_script="src/models/train_ml_models.py",
        test_script="src/models/inference/predict_rf.py",
        best_epoch="",
        best_valid_auc=rf_mean_auc,
        test_mean_auc=rf_mean_auc,
        learning_rate="",
        batch_size="",
        num_epochs="",
        notes="auto-logged from train_ml_models.py after offline training and evaluation",
    )
    print(f"Experiment record saved: {exp_id}")


if __name__ == "__main__":
    main()