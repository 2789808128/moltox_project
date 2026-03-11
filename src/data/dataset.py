import pandas as pd
import torch
from torch.utils.data import Dataset


class Tox21Dataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        """
        Args:
            csv_path: 数据文件路径，如 tox21_train.csv
            tokenizer: 你实现的 SmilesTokenizer
            max_length: 最大序列长度
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 固定 12 个任务标签列
        self.label_cols = [
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

        # 检查标签列是否都存在
        for col in self.label_cols:
            if col not in self.df.columns:
                raise ValueError(f"标签列缺失: {col}")

        if "smiles" not in self.df.columns:
            raise ValueError("数据中缺少 smiles 列")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        smiles = row["smiles"]

        # tokenizer 编码
        encoded = self.tokenizer.encode(smiles, max_length=self.max_length)

        # 处理 labels 和 label_mask
        labels = []
        label_mask = []

        for col in self.label_cols:
            value = row[col]

            if pd.isna(value):
                labels.append(0.0)       # 占位值
                label_mask.append(0.0)   # 表示该标签无效
            else:
                labels.append(float(value))
                label_mask.append(1.0)   # 表示该标签有效

        sample = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
            "label_mask": torch.tensor(label_mask, dtype=torch.float),
        }

        return sample
