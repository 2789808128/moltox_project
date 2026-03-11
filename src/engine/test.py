import os
import torch
import subprocess
import sys

from src.data.build_dataloader import build_dataloaders
from src.models.transformer_model import SmilesTransformer
from src.engine.losses import MaskedBCEWithLogitsLoss
from src.engine.evaluate import evaluate
from src.utils.experiment_logger import append_experiment_record


def main():
    project_root = r"E:\Project\moltox_project"

    train_csv = os.path.join(project_root, "data", "processed", "tox21_train.csv")
    valid_csv = os.path.join(project_root, "data", "processed", "tox21_valid.csv")
    test_csv = os.path.join(project_root, "data", "processed", "tox21_test.csv")

    checkpoint_path = os.path.join(
        project_root,
        "outputs",
        "checkpoints",
        "transformer",
        "best_smiles_transformer.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    max_length = checkpoint["max_length"]

    print("\nLoaded checkpoint info:")
    print("Saved epoch:", checkpoint["epoch"])
    print("Best epoch:", checkpoint.get("best_epoch", checkpoint["epoch"]))
    print("Best valid ROC-AUC:", checkpoint["best_valid_auc"])

    tokenizer, train_loader, valid_loader, test_loader = build_dataloaders(
        train_csv_path=train_csv,
        valid_csv_path=valid_csv,
        test_csv_path=test_csv,
        max_length=max_length,
        batch_size=32,
        num_workers=0,
    )

    print("Tokenizer vocab size:", tokenizer.vocab_size())
    print("Test batches:", len(test_loader))

    model = SmilesTransformer(
        vocab_size=tokenizer.vocab_size(),
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"],
        max_len=max_length,
        num_tasks=model_config["num_tasks"],
        pad_token_id=model_config["pad_token_id"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = MaskedBCEWithLogitsLoss()

    test_results = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    print("\n" + "=" * 60)
    print("Transformer Test Results")
    print("=" * 60)
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Mean ROC-AUC: {test_results['mean_auc']:.4f}")

    print("\nPer-task Test ROC-AUC:")
    for task_name, auc in test_results["task_auc_dict"].items():
        if auc is None:
            print(f"{task_name}: None")
        else:
            print(f"{task_name}: {auc:.4f}")

    exp_id = append_experiment_record(
        project_root=project_root,
        model_type="transformer",
        stage="test",
        train_script="src/engine/train.py",
        test_script="src/engine/test.py",
        best_epoch=checkpoint.get("best_epoch", checkpoint.get("epoch", "")),
        best_valid_auc=checkpoint.get("best_valid_auc", ""),
        test_mean_auc=test_results["mean_auc"],
        learning_rate="",
        batch_size=32,
        num_epochs="",
        notes="auto-logged from transformer test script",
    )

    print(f"Experiment record saved: {exp_id}")

    print("\nRefreshing frontend assets...")
    refresh_script = os.path.join(project_root, "src", "utils", "refresh_frontend_assets.py")
    subprocess.run([sys.executable, refresh_script], check=True)


if __name__ == "__main__":
    main()