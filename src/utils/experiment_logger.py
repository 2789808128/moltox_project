from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


DEFAULT_FIELDS = [
    "experiment_id",
    "model_type",
    "run_date",
    "stage",
    "train_script",
    "test_script",
    "best_epoch",
    "best_valid_auc",
    "test_mean_auc",
    "learning_rate",
    "batch_size",
    "num_epochs",
    "notes",
]


def ensure_experiment_registry(project_root: str | Path) -> Path:
    project_root = Path(project_root)
    exp_dir = project_root / "outputs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    registry_path = exp_dir / "experiment_registry.csv"

    if not registry_path.exists():
        with registry_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=DEFAULT_FIELDS)
            writer.writeheader()

    return registry_path


def get_next_experiment_id(registry_path: str | Path) -> str:
    registry_path = Path(registry_path)

    if not registry_path.exists():
        return "exp_001"

    with registry_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return "exp_001"

    max_num = 0
    for row in rows:
        exp_id = row.get("experiment_id", "")
        if exp_id.startswith("exp_"):
            try:
                num = int(exp_id.split("_")[1])
                max_num = max(max_num, num)
            except Exception:
                pass

    return f"exp_{max_num + 1:03d}"


def append_experiment_record(
    project_root: str | Path,
    model_type: str,
    stage: str,
    train_script: str = "",
    test_script: str = "",
    best_epoch: int | str | None = None,
    best_valid_auc: float | str | None = None,
    test_mean_auc: float | str | None = None,
    learning_rate: float | str | None = None,
    batch_size: int | str | None = None,
    num_epochs: int | str | None = None,
    notes: str = "",
) -> str:
    registry_path = ensure_experiment_registry(project_root)
    experiment_id = get_next_experiment_id(registry_path)

    row = {
        "experiment_id": experiment_id,
        "model_type": model_type,
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "train_script": train_script,
        "test_script": test_script,
        "best_epoch": "" if best_epoch is None else best_epoch,
        "best_valid_auc": "" if best_valid_auc is None else best_valid_auc,
        "test_mean_auc": "" if test_mean_auc is None else test_mean_auc,
        "learning_rate": "" if learning_rate is None else learning_rate,
        "batch_size": "" if batch_size is None else batch_size,
        "num_epochs": "" if num_epochs is None else num_epochs,
        "notes": notes,
    }

    with registry_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_FIELDS)
        writer.writerow(row)

    return experiment_id