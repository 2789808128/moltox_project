import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class SmilesTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
        num_tasks: int = 12,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # 1. token embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id
        )

        # 2. positional encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # 3. transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # 4. dropout + classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_tasks)

    def masked_mean_pooling(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        attention_mask: [batch_size, seq_len]
        return: [batch_size, d_model]
        """
        mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        x = x * mask

        sum_x = x.sum(dim=1)  # [batch_size, d_model]
        valid_token_count = mask.sum(dim=1).clamp(min=1e-9)  # [batch_size, 1]

        pooled = sum_x / valid_token_count
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        return: logits [batch_size, num_tasks]
        """
        # padding mask: Transformer 里 True 表示这个位置要被忽略
        src_key_padding_mask = (input_ids == self.pad_token_id)  # [batch_size, seq_len]

        # 1. embedding
        x = self.embedding(input_ids)  # [batch_size, seq_len, d_model]

        # 常见做法：乘 sqrt(d_model)，稳定表示尺度
        x = x * math.sqrt(self.d_model)

        # 2. add positional encoding
        x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]

        # 3. transformer encoder
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # [batch_size, seq_len, d_model]

        # 4. mean pooling over valid tokens
        pooled = self.masked_mean_pooling(x, attention_mask)  # [batch_size, d_model]

        # 5. classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [batch_size, num_tasks]

        return logits

if __name__ == "__main__":
    from src.data.build_dataloader import build_dataloaders

    train_csv = r"E:\Project\moltox_project\data\processed\tox21_train.csv"
    valid_csv = r"E:\Project\moltox_project\data\processed\tox21_valid.csv"
    test_csv = r"E:\Project\moltox_project\data\processed\tox21_test.csv"

    tokenizer, train_loader, valid_loader, test_loader = build_dataloaders(
        train_csv_path=train_csv,
        valid_csv_path=valid_csv,
        test_csv_path=test_csv,
        max_length=64,
        batch_size=4,
        num_workers=0,
    )

    model = SmilesTransformer(
        vocab_size=tokenizer.vocab_size(),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=64,
        num_tasks=12,
        pad_token_id=0,
    )

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    logits = model(input_ids=input_ids, attention_mask=attention_mask)

    print("input_ids shape:", input_ids.shape)
    print("attention_mask shape:", attention_mask.shape)
    print("logits shape:", logits.shape)
    print("\nlogits:")
    print(logits)