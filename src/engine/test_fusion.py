import os
import torch

from src.data.build_dataloader_fusion import build_dataloaders_fusion
from src.models.fusion_model import SmilesMorganFusionModel
from src.engine.losses import MaskedBCEWithLogitsLoss
from src.engine.metrics import compute_multitask_roc_auc


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

    import numpy as np

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
    # =========================
    # 1. 路径
    # =========================
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

    # =========================
    # 2. 设备
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # 3. 数据
    # tokenizer 仍然只用训练集构建
    # =========================
    tokenizer, train_loader, valid_loader, test_loader = build_dataloaders_fusion(
        train_csv_path=train_csv,
        valid_csv_path=valid_csv,
        test_csv_path=test_csv,
        max_length=64,
        batch_size=32,
        num_workers=0,
        fp_radius=2,
        fp_n_bits=2048,
    )

    print("Tokenizer vocab size:", tokenizer.vocab_size())
    print("Test batches:", len(test_loader))

    # =========================
    # 4. 加载 checkpoint
    # =========================
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("\nLoaded fusion checkpoint info:")
    print("Saved epoch:", checkpoint["epoch"])
    print("Best valid ROC-AUC:", checkpoint["best_valid_auc"])

    # =========================
    # 5. 重建模型
    # 参数要与训练时一致
    # =========================
    model = SmilesMorganFusionModel(
        vocab_size=tokenizer.vocab_size(),
        fp_dim=2048,
        d_model=128,
        fp_hidden_dim=256,
        fusion_hidden_dim=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=64,
        num_tasks=12,
        pad_token_id=0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    # =========================
    # 6. 损失函数
    # =========================
    criterion = MaskedBCEWithLogitsLoss()

    # =========================
    # 7. 测试集评估
    # =========================
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


if __name__ == "__main__":
    main()