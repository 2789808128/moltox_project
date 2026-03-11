from __future__ import annotations

import csv
import shutil
from pathlib import Path

from src.utils.paths import get_project_root
from src.utils.plot_model_comparison import plot_mean_auc_comparison
from src.utils.plot_task_auc_comparison import plot_task_auc_comparison


def copy_file(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")


def read_latest_test_auc_map(registry_csv: Path) -> dict:
    """
    从 experiment_registry.csv 中读取每个模型最新一条 test 记录的 test_mean_auc
    返回:
    {
        "transformer": 0.7554,
        "morgan_logreg": 0.7858,
        "morgan_rf": 0.8131,
        "fusion": 0.8026,
    }
    """
    if not registry_csv.exists():
        raise FileNotFoundError(f"Experiment registry not found: {registry_csv}")

    latest = {}

    with registry_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("stage") != "test":
                continue

            model_type = row.get("model_type", "").strip()
            test_mean_auc = row.get("test_mean_auc", "").strip()

            if not model_type or not test_mean_auc:
                continue

            try:
                latest[model_type] = float(test_mean_auc)
            except ValueError:
                continue

    return latest


def refresh_frontend_assets():
    project_root = get_project_root()

    outputs_dir = project_root / "outputs"
    logs_dir = outputs_dir / "logs"
    predictions_dir = outputs_dir / "predictions"
    experiments_dir = outputs_dir / "experiments"

    # 你的前端静态资源目录
    frontend_public_dir = project_root / "frontend" / "public"
    frontend_public_dir.mkdir(parents=True, exist_ok=True)

    registry_csv = experiments_dir / "experiment_registry.csv"

    # 1. 从实验记录里读最新 test auc，重画总体模型对比图
    auc_map = read_latest_test_auc_map(registry_csv)

    required_models = ["transformer", "morgan_logreg", "morgan_rf", "fusion"]
    missing = [m for m in required_models if m not in auc_map]
    if missing:
        raise ValueError(f"Missing latest test_mean_auc for models: {missing}")

    model_comparison_path = predictions_dir / "model_comparison_mean_auc.png"
    plot_mean_auc_comparison(
        save_path=str(model_comparison_path),
        transformer_auc=auc_map["transformer"],
        logreg_auc=auc_map["morgan_logreg"],
        rf_auc=auc_map["morgan_rf"],
        fusion_auc=auc_map["fusion"],
    )

    # 2. 重画任务级对比图
    # 当前版本仍使用 plot_task_auc_comparison.py 内部的数据
    task_auc_path = predictions_dir / "task_auc_comparison.png"
    plot_task_auc_comparison(save_path=str(task_auc_path))

    # 3. 从 logs 中取最新训练曲线
    transformer_curve_src = logs_dir / "transformer" / "transformer_curves.png"
    fusion_curve_src = logs_dir / "fusion" / "fusion_curves.png"

    # 4. 同步到前端 public 根目录
    copy_file(transformer_curve_src, frontend_public_dir / "transformer_curves.png")
    copy_file(fusion_curve_src, frontend_public_dir / "fusion_curves.png")
    copy_file(model_comparison_path, frontend_public_dir / "model_comparison_mean_auc.png")
    copy_file(task_auc_path, frontend_public_dir / "task_auc_comparison.png")

    print("\nFrontend assets refreshed successfully.")


if __name__ == "__main__":
    refresh_frontend_assets()