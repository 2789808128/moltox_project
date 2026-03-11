import torch
import torch.nn as nn


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 不要直接求平均，先保留每个位置的 loss
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        logits: [batch_size, num_tasks]
        labels: [batch_size, num_tasks]
        label_mask: [batch_size, num_tasks], 有效标签为1，缺失标签为0
        """
        # 逐元素 BCE loss
        loss_matrix = self.bce(logits, labels)  # [batch_size, num_tasks]

        # 只保留有效标签位置
        masked_loss = loss_matrix * label_mask  # [batch_size, num_tasks]

        # 有效标签总数
        valid_count = label_mask.sum().clamp(min=1e-8)

        # 只对有效位置求平均
        loss = masked_loss.sum() / valid_count

        return loss
if __name__ == "__main__":
    criterion = MaskedBCEWithLogitsLoss()

    # 假设 batch_size=2, num_tasks=3
    logits = torch.tensor([
        [0.2, -1.0, 0.5],
        [1.2,  0.3, -0.7]
    ], dtype=torch.float)

    labels = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ], dtype=torch.float)

    label_mask = torch.tensor([
        [1.0, 1.0, 0.0],   # 第3个任务缺失
        [1.0, 0.0, 1.0]    # 第2个任务缺失
    ], dtype=torch.float)

    loss = criterion(logits, labels, label_mask)

    print("logits shape:", logits.shape)
    print("labels shape:", labels.shape)
    print("label_mask shape:", label_mask.shape)
    print("loss:", loss.item())