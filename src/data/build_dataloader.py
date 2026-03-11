import pandas as pd
from torch.utils.data import DataLoader

from src.models.tokenizer import SmilesTokenizer
from src.data.dataset import Tox21Dataset


def build_tokenizer(train_csv_path: str):
    """
    只使用训练集构建 tokenizer 词表
    """
    df_train = pd.read_csv(train_csv_path)
    train_smiles = df_train["smiles"].tolist()

    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(train_smiles)

    return tokenizer


def build_dataloaders(
    train_csv_path: str,
    valid_csv_path: str,
    test_csv_path: str,
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    构建 tokenizer、train/valid/test dataset 和 dataloader
    """
    # 1. tokenizer 只用训练集构建
    tokenizer = build_tokenizer(train_csv_path)

    # 2. dataset
    train_dataset = Tox21Dataset(
        csv_path=train_csv_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    valid_dataset = Tox21Dataset(
        csv_path=valid_csv_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = Tox21Dataset(
        csv_path=test_csv_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # 3. dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return tokenizer, train_loader, valid_loader, test_loader
