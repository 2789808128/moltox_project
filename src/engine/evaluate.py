import torch
import numpy as np

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
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0

    all_labels = []
    all_label_masks = []
    all_probs = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_mask"].to(device)

        # 前向传播
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # loss
        loss = criterion(logits, labels, label_mask)
        total_loss += loss.item()

        # logits -> probabilities
        probs = torch.sigmoid(logits)

        # 收集到 CPU / numpy
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
if __name__ == "__main__":
    from src.data.build_dataloader import build_dataloaders
    from src.models.transformer_model import SmilesTransformer
    from src.engine.losses import MaskedBCEWithLogitsLoss

    train_csv = r"E:\Project\moltox_project\data\processed\tox21_train.csv"
    valid_csv = r"E:\Project\moltox_project\data\processed\tox21_valid.csv"
    test_csv = r"E:\Project\moltox_project\data\processed\tox21_test.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer, train_loader, valid_loader, test_loader = build_dataloaders(
        train_csv_path=train_csv,
        valid_csv_path=valid_csv,
        test_csv_path=test_csv,
        max_length=64,
        batch_size=32,
        num_workers=0,
    )

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

    criterion = MaskedBCEWithLogitsLoss()

    results = evaluate(
        model=model,
        dataloader=valid_loader,
        criterion=criterion,
        device=device
    )

    print("\nValidation Loss:", results["loss"])
    print("Validation Mean ROC-AUC:", results["mean_auc"])
    print("\nPer-task ROC-AUC:")
    for task_name, auc in results["task_auc_dict"].items():
        print(f"{task_name}: {auc}")