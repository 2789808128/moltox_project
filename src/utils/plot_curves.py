import os
import pandas as pd
import matplotlib.pyplot as plt


def save_history_to_csv(history, csv_path):
    """
    history: list of dict
    每个元素类似:
    {
        "epoch": 1,
        "train_loss": 0.3212,
        "valid_loss": 0.2414,
        "valid_mean_auc": 0.6619
    }
    """
    df = pd.DataFrame(history)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"History CSV saved to: {csv_path}")


def plot_training_curves(history, save_path, title="Training Curves"):
    """
    将训练历史画成 3 张子图：
    - train loss
    - valid loss
    - valid mean auc
    """
    df = pd.DataFrame(history)

    epochs = df["epoch"].tolist()
    train_loss = df["train_loss"].tolist()
    valid_loss = df["valid_loss"].tolist()
    valid_auc = df["valid_mean_auc"].tolist()

    plt.figure(figsize=(12, 4))

    # 1. train loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss")

    # 2. valid loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, valid_loss, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Valid Loss")
    plt.title("Valid Loss")

    # 3. valid mean auc
    plt.subplot(1, 3, 3)
    plt.plot(epochs, valid_auc, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Valid Mean ROC-AUC")
    plt.title("Valid Mean ROC-AUC")

    plt.suptitle(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Training curves saved to: {save_path}")