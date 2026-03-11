import os
import matplotlib.pyplot as plt


def plot_mean_auc_comparison(
    save_path: str,
    transformer_auc: float = 0.7554,
    logreg_auc: float = 0.7858,
    rf_auc: float = 0.8131,
    fusion_auc: float = 0.8026,
):
    model_names = [
        "Transformer",
        "Morgan+LogReg",
        "Morgan+RF",
        "Fusion"
    ]

    mean_aucs = [
        transformer_auc,
        logreg_auc,
        rf_auc,
        fusion_auc,
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, mean_aucs)

    plt.ylabel("Test Mean ROC-AUC")
    plt.title("Tox21 Model Comparison")
    plt.ylim(0.70, 0.85)

    for bar, value in zip(bars, mean_aucs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Comparison figure saved to: {save_path}")


def main():
    project_root = r"E:\Project\moltox_project"
    save_path = os.path.join(
        project_root,
        "outputs",
        "predictions",
        "model_comparison_mean_auc.png"
    )

    plot_mean_auc_comparison(save_path)


if __name__ == "__main__":
    main()