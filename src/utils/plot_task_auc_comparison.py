import os
import numpy as np
import matplotlib.pyplot as plt


def plot_task_auc_comparison(save_path: str):
    task_names = [
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

    transformer_auc = [
        0.8109, 0.9003, 0.8223, 0.6604,
        0.7479, 0.7800, 0.5854, 0.7404,
        0.8035, 0.6770, 0.7857, 0.7514
    ]

    logreg_auc = [
        0.7451, 0.9370, 0.8468, 0.7341,
        0.6770, 0.7762, 0.7811, 0.7374,
        0.7287, 0.7397, 0.8396, 0.8873
    ]

    rf_auc = [
        0.7499, 0.8685, 0.8837, 0.7599,
        0.7558, 0.8182, 0.7986, 0.7849,
        0.8071, 0.7691, 0.8578, 0.9042
    ]

    fusion_auc = [
        0.7733, 0.9406, 0.8363, 0.7962,
        0.7596, 0.8261, 0.6464, 0.7430,
        0.7896, 0.7660, 0.8567, 0.8355
    ]

    x = np.arange(len(task_names))
    width = 0.2

    plt.figure(figsize=(16, 6))

    plt.bar(x - 1.5 * width, transformer_auc, width, label="Transformer")
    plt.bar(x - 0.5 * width, logreg_auc, width, label="Morgan+LogReg")
    plt.bar(x + 0.5 * width, rf_auc, width, label="Morgan+RF")
    plt.bar(x + 1.5 * width, fusion_auc, width, label="Fusion")

    plt.xticks(x, task_names, rotation=45, ha="right")
    plt.ylabel("Test ROC-AUC")
    plt.title("Per-task ROC-AUC Comparison on Tox21")
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Task-level comparison figure saved to: {save_path}")


def main():
    project_root = r"E:\Project\moltox_project"
    save_path = os.path.join(
        project_root,
        "outputs",
        "predictions",
        "task_auc_comparison.png"
    )

    plot_task_auc_comparison(save_path)


if __name__ == "__main__":
    main()