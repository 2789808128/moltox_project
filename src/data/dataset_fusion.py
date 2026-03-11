import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class Tox21FusionDataset(Dataset):
    def __init__(
        self,
        csv_path,
        tokenizer,
        max_length=128,
        fp_radius=2,
        fp_n_bits=2048
    ):
        """
        Args:
            csv_path: 数据文件路径
            tokenizer: 你实现的 SmilesTokenizer
            max_length: SMILES 最大序列长度
            fp_radius: Morgan fingerprint 半径
            fp_n_bits: Morgan fingerprint 维度
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fp_radius = fp_radius
        self.fp_n_bits = fp_n_bits

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

        for col in self.label_cols:
            if col not in self.df.columns:
                raise ValueError(f"标签列缺失: {col}")

        if "smiles" not in self.df.columns:
            raise ValueError("数据中缺少 smiles 列")

        # 使用新版 MorganGenerator，避免 deprecated warning
        self.morgan_generator = GetMorganGenerator(
            radius=self.fp_radius,
            fpSize=self.fp_n_bits
        )

        # 一次性预计算所有 fingerprint，避免 __getitem__ 中重复计算
        print(f"Precomputing Morgan fingerprints for: {csv_path}")
        self.fingerprints = self._precompute_fingerprints()

    def __len__(self):
        return len(self.df)

    def _smiles_to_morgan_fp(self, smiles: str):
        """
        将单条 SMILES 转成 Morgan fingerprint
        返回 shape = [fp_n_bits] 的 numpy 向量
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 正常情况下不会发生，因为你已经清洗过数据
            return np.zeros((self.fp_n_bits,), dtype=np.float32)

        fp = self.morgan_generator.GetFingerprint(mol)

        arr = np.zeros((self.fp_n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _precompute_fingerprints(self):
        """
        预计算整个数据集的 Morgan fingerprint
        """
        fps = []
        for smiles in self.df["smiles"].tolist():
            fp = self._smiles_to_morgan_fp(smiles)
            fps.append(fp)

        return np.array(fps, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        smiles = row["smiles"]

        # 1. 序列编码
        encoded = self.tokenizer.encode(smiles, max_length=self.max_length)

        # 2. 直接取预计算 fingerprint
        fingerprint = self.fingerprints[idx]

        # 3. labels 与 label_mask
        labels = []
        label_mask = []

        for col in self.label_cols:
            value = row[col]

            if pd.isna(value):
                labels.append(0.0)
                label_mask.append(0.0)
            else:
                labels.append(float(value))
                label_mask.append(1.0)

        sample = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "fingerprint": torch.tensor(fingerprint, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.float),
            "label_mask": torch.tensor(label_mask, dtype=torch.float),
        }

        return sample


if __name__ == "__main__":
    from src.models.tokenizer import SmilesTokenizer

    train_csv = r"E:\Project\moltox_project\data\processed\tox21_train.csv"

    # 用训练集构建 tokenizer
    df = pd.read_csv(train_csv)
    train_smiles = df["smiles"].tolist()

    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(train_smiles)

    dataset = Tox21FusionDataset(
        csv_path=train_csv,
        tokenizer=tokenizer,
        max_length=64,
        fp_radius=2,
        fp_n_bits=2048
    )

    print("数据集大小:", len(dataset))

    sample = dataset[0]

    print("\n样本字段:")
    print(sample.keys())

    print("\ninput_ids shape:", sample["input_ids"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)
    print("fingerprint shape:", sample["fingerprint"].shape)
    print("labels shape:", sample["labels"].shape)
    print("label_mask shape:", sample["label_mask"].shape)

    print("\n前10维 fingerprint:")
    print(sample["fingerprint"][:10])

    print("\nlabels:")
    print(sample["labels"])

    print("\nlabel_mask:")
    print(sample["label_mask"])