import os
import torch

from src.data.build_dataloader import build_dataloaders
from src.models.transformer_model import SmilesTransformer
from src.engine.losses import MaskedBCEWithLogitsLoss
from src.engine.evaluate import evaluate


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
        "best_smiles_transformer.pt"
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
    tokenizer, train_loader, valid_loader, test_loader = build_dataloaders(
        train_csv_path=train_csv,
        valid_csv_path=valid_csv,
        test_csv_path=test_csv,
        max_length=64,
        batch_size=32,
        num_workers=0,
    )

    print("Tokenizer vocab size:", tokenizer.vocab_size())
    print("Test batches:", len(test_loader))

    # =========================
    # 4. 加载 checkpoint
    # =========================
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("\nLoaded checkpoint info:")
    print("Saved epoch:", checkpoint["epoch"])
    print("Best valid ROC-AUC:", checkpoint["best_valid_auc"])

    # =========================
    # 5. 重建模型
    # 参数要与训练时一致
    # =========================
    model = SmilesTransformer(
        vocab_size=tokenizer.vocab_size(),
        d_model=128,
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
    test_results = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Mean ROC-AUC: {test_results['mean_auc']:.4f}")

    print("\nPer-task Test ROC-AUC:")
    for task_name, auc in test_results["task_auc_dict"].items():
        if auc is None:
            print(f"{task_name}: None")
        else:
            print(f"{task_name}: {auc:.4f}")


if __name__ == "__main__":
    main()