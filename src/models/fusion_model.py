import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class SmilesMorganFusionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        fp_dim: int = 2048,
        d_model: int = 128,
        fp_hidden_dim: int = 256,
        fusion_hidden_dim: int = 128,
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

        # =========================
        # 1. SMILES Transformer 分支
        # =========================
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id
        )

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

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

        # =========================
        # 2. Morgan Fingerprint 分支
        # =========================
        self.fp_mlp = nn.Sequential(
            nn.Linear(fp_dim, fp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fp_hidden_dim, fp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # =========================
        # 3. 融合分类头
        # smiles_repr: d_model
        # fp_repr: fp_hidden_dim
        # concat dim = d_model + fp_hidden_dim
        # =========================
        fusion_input_dim = d_model + fp_hidden_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_tasks)
        )

    def masked_mean_pooling(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        attention_mask: [batch_size, seq_len]
        return: [batch_size, d_model]
        """
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        x = x * mask

        sum_x = x.sum(dim=1)  # [B, d_model]
        valid_token_count = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]

        pooled = sum_x / valid_token_count
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fingerprint: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        fingerprint: [B, fp_dim]
        return: logits [B, num_tasks]
        """
        # =========================
        # 1. Transformer 分支
        # =========================
        src_key_padding_mask = (input_ids == self.pad_token_id)

        x = self.embedding(input_ids)  # [B, L, d_model]
        x = x * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # [B, L, d_model]

        smiles_repr = self.masked_mean_pooling(x, attention_mask)  # [B, d_model]

        # =========================
        # 2. Fingerprint 分支
        # =========================
        fp_repr = self.fp_mlp(fingerprint)  # [B, fp_hidden_dim]

        # =========================
        # 3. 融合
        # =========================
        fused_repr = torch.cat([smiles_repr, fp_repr], dim=-1)  # [B, d_model + fp_hidden_dim]

        logits = self.fusion_head(fused_repr)  # [B, num_tasks]

        return logits
