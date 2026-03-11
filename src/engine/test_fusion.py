import os
import torch
import numpy as np
import subprocess
import sys

from src.data.build_dataloader_fusion import build_dataloaders_fusion
from src.models.fusion_model import SmilesMorganFusionModel
from src.engine.losses import MaskedBCEWithLogitsLoss
from src.engine.metrics import compute_multitask_roc_auc
from src.utils.experiment_logger import append_experiment_record


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


@torch.no_grad()
def evaluate_fusion(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_label_masks = []
    all_probs = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        fingerprint = batch["fingerprint"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_mask"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fingerprint=fingerprint
        )

        loss = criterion(logits, labels, label_mask)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        all_labels.append(labels.cpu().numpy())
        all_label_masks.append(label_mask.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    all_labels = np.concatenate(all_labels, axis=0)
    all_label_masks = np.concatenate(all_label_masks, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    task_auc_dict, mean_auc = compute_multitask_roc_auc(
        labels=all_labels,
        probs=all_probs,
        label_mask=all_label_masks,
        task_names=TASK_NAMES
    )

    return {
        "loss": avg_loss,
        "task_auc_dict": task_auc_dict,
        "mean_auc": mean_auc
    }


def main():
    project_root = r"E:\Project\moltox_project"

    train_csv = os.path.join(project_root, "data", "processed", "tox21_train.csv")
    valid_csv = os.path.join(project_root, "data", "processed", "tox21_valid.csv")
    test_csv = os.path.join(project_root, "data", "processed", "tox21_test.csv")

    checkpoint_path = os.path.join(
        project_root,
        "outputs",
        "checkpoints",
        "fusion",
        "best_fusion_model.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    max_length = checkpoint["max_length"]
    fp_n_bits = checkpoint["fp_n_bits"]

    print("\nLoaded fusion checkpoint info:")
    print("Saved epoch:", checkpoint["epoch"])
    print("Best epoch:", checkpoint.get("best_epoch", checkpoint["epoch"]))
    print("Best valid ROC-AUC:", checkpoint["best_valid_auc"])

    tokenizer, train_loader, valid_loader, test_loader = build_dataloaders_fusion(
        train_csv_path=train_csv,
        valid_csv_path=valid_csv,
        test_csv_path=test_csv,
        max_length=max_length,
        batch_size=32,
        num_workers=0,
        fp_radius=2,
        fp_n_bits=fp_n_bits,
    )

    print("Tokenizer vocab size:", tokenizer.vocab_size())
    print("Test batches:", len(test_loader))

    model = SmilesMorganFusionModel(
        vocab_size=tokenizer.vocab_size(),
        fp_dim=fp_n_bits,
        d_model=model_config["d_model"],
        fp_hidden_dim=model_config["fp_hidden_dim"],
        fusion_hidden_dim=model_config["fusion_hidden_dim"],
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

    test_results = evaluate_fusion(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    print("\n" + "=" * 60)
    print("Fusion Test Results")
    print("=" * 60)
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Mean ROC-AUC: {test_results['mean_auc']:.4f}")

    print("\nPer-task Fusion Test ROC-AUC:")
    for task_name, auc in test_results["task_auc_dict"].items():
        if auc is None:
            print(f"{task_name}: None")
        else:
            print(f"{task_name}: {auc:.4f}")

    exp_id = append_experiment_record(
        project_root=project_root,
        model_type="fusion",
        stage="test",
        train_script="src/engine/train_fusion.py",
        test_script="src/engine/test_fusion.py",
        best_epoch=checkpoint.get("best_epoch", checkpoint.get("epoch", "")),
        best_valid_auc=checkpoint.get("best_valid_auc", ""),
        test_mean_auc=test_results["mean_auc"],
        learning_rate="",
        batch_size=32,
        num_epochs="",
        notes="auto-logged from fusion test script",
    )

    print(f"Experiment record saved: {exp_id}")

    print("\nRefreshing frontend assets...")
    refresh_script = os.path.join(project_root, "src", "utils", "refresh_frontend_assets.py")
    subprocess.run([sys.executable, refresh_script], check=True)


if __name__ == "__main__":
    main()