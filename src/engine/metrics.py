import numpy as np
from sklearn.metrics import roc_auc_score


def compute_task_roc_auc(y_true, y_prob):
    """
    计算单个任务的 ROC-AUC
    如果该任务标签全是同一类，则返回 None
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return None

    return roc_auc_score(y_true, y_prob)


def compute_multitask_roc_auc(labels, probs, label_mask, task_names):
    """
    计算多任务 ROC-AUC

    Args:
        labels: [num_samples, num_tasks]
        probs: [num_samples, num_tasks]
        label_mask: [num_samples, num_tasks]
        task_names: 任务名称列表

    Returns:
        task_auc_dict: 每个任务的 AUC 字典
        mean_auc: 所有有效任务的平均 AUC
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    label_mask = np.asarray(label_mask)

    task_auc_dict = {}
    valid_aucs = []

    num_tasks = labels.shape[1]

    for task_idx in range(num_tasks):
        task_name = task_names[task_idx]

        # 只保留该任务有效标签的位置
        valid_indices = label_mask[:, task_idx] == 1

        task_labels = labels[valid_indices, task_idx]
        task_probs = probs[valid_indices, task_idx]

        auc = compute_task_roc_auc(task_labels, task_probs)
        task_auc_dict[task_name] = auc

        if auc is not None:
            valid_aucs.append(auc)

    mean_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else None

    return task_auc_dict, mean_auc
