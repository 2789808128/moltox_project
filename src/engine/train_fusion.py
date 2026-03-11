import os
import torch
import torch.optim as optim
import numpy as np

from src.data.build_dataloader_fusion import build_dataloaders_fusion
from src.models.fusion_model import SmilesMorganFusionModel
from src.engine.losses import MaskedBCEWithLogitsLoss
from src.engine.metrics import compute_multitask_roc_auc
from src.utils.plot_curves import save_history_to_csv, plot_training_curves
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


def train_one_epoch_fusion(model, dataloader, criterion, optimizer, device, grad_clip=None):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        fingerprint = batch["fingerprint"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_mask"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fingerprint=fingerprint
        )

        loss = criterion(logits, labels, label_mask)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            print(f"  Batch [{batch_idx + 1}/{len(dataloader)}] - Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


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

    checkpoint_dir = os.path.join(project_root, "outputs", "checkpoints", "fusion")
    log_dir = os.path.join(project_root, "outputs", "logs", "fusion")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, "best_fusion_model.pt")
    history_csv_path = os.path.join(log_dir, "fusion_history.csv")
    curves_png_path = os.path.join(log_dir, "fusion_curves.png")

    # -------------------------
    # Config
    # -------------------------
    max_length = 64
    batch_size = 32
    num_workers = 0

    fp_radius = 2
    fp_n_bits = 2048

    d_model = 128
    fp_hidden_dim = 256
    fusion_hidden_dim = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    num_tasks = 12

    lr = 1e-4
    num_epochs = 20

    # 新增
    early_stopping_patience = 5
    scheduler_patience = 2
    scheduler_factor = 0.5
    min_lr = 1e-6
    grad_clip = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer, train_loader, valid_loader, test_loader = build_dataloaders_fusion(
        train_csv_path=train_csv,
        valid_csv_path=valid_csv,
        test_csv_path=test_csv,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
        fp_radius=fp_radius,
        fp_n_bits=fp_n_bits,
    )

    print("Tokenizer vocab size:", tokenizer.vocab_size())
    print("Train batches:", len(train_loader))
    print("Valid batches:", len(valid_loader))
    print("Test batches:", len(test_loader))

    model = SmilesMorganFusionModel(
        vocab_size=tokenizer.vocab_size(),
        fp_dim=fp_n_bits,
        d_model=d_model,
        fp_hidden_dim=fp_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_length,
        num_tasks=num_tasks,
        pad_token_id=0,
    ).to(device)

    criterion = MaskedBCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )

    best_valid_auc = -1.0
    best_epoch = -1
    no_improve_count = 0
    history = []

    for epoch in range(num_epochs):
        print("\n" + "=" * 60)
        print(f"Fusion Epoch [{epoch + 1}/{num_epochs}]")
        print("=" * 60)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.6f}")

        train_loss = train_one_epoch_fusion(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip
        )

        valid_results = evaluate_fusion(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device
        )

        valid_loss = valid_results["loss"]
        valid_mean_auc = valid_results["mean_auc"]

        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_mean_auc": valid_mean_auc,
        })

        print(f"\nFusion Epoch [{epoch + 1}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}")
        print(f"  Valid Mean ROC-AUC: {valid_mean_auc:.4f}")

        if valid_mean_auc is not None:
            scheduler.step(valid_mean_auc)

        if valid_mean_auc is not None and valid_mean_auc > best_valid_auc:
            best_valid_auc = valid_mean_auc
            best_epoch = epoch + 1
            no_improve_count = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_valid_auc": best_valid_auc,
                    "best_epoch": best_epoch,
                    "vocab_size": tokenizer.vocab_size(),
                    "max_length": max_length,
                    "fp_n_bits": fp_n_bits,
                    "model_config": {
                        "d_model": d_model,
                        "fp_hidden_dim": fp_hidden_dim,
                        "fusion_hidden_dim": fusion_hidden_dim,
                        "nhead": nhead,
                        "num_layers": num_layers,
                        "dim_feedforward": dim_feedforward,
                        "dropout": dropout,
                        "num_tasks": num_tasks,
                        "pad_token_id": 0,
                    }
                },
                best_model_path
            )

            print(f"  Best fusion model saved to: {best_model_path}")
            print(f"  Updated best valid ROC-AUC: {best_valid_auc:.4f}")
            print(f"  Best epoch so far: {best_epoch}")
        else:
            no_improve_count += 1
            print(f"  No improvement count: {no_improve_count}/{early_stopping_patience}")

        if no_improve_count >= early_stopping_patience:
            print("\nEarly stopping triggered.")
            break

    save_history_to_csv(history, history_csv_path)
    plot_training_curves(
        history,
        save_path=curves_png_path,
        title="Fusion Training Curves"
    )

    print("\nFusion training finished.")
    print(f"Best Valid ROC-AUC: {best_valid_auc:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best fusion model path: {best_model_path}")
    print(f"History saved to: {history_csv_path}")
    print(f"Curves saved to: {curves_png_path}")
    exp_id = append_experiment_record(
        project_root=project_root,
        model_type="fusion",
        stage="train",
        train_script="src/engine/train_fusion.py",
        test_script="src/engine/test_fusion.py",
        best_epoch=best_epoch,
        best_valid_auc=best_valid_auc,
        learning_rate=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        notes="round1 fusion training upgrade with scheduler, early stopping, grad clip",
    )

    print(f"Experiment record saved: {exp_id}")


if __name__ == "__main__":
    main()