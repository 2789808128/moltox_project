import json
from typing import List, Dict


class SmilesTokenizer:
    def __init__(self):
        # 特殊 token
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"

        # 固定特殊 token 的 id
        self.token_to_id: Dict[str, int] = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.cls_token: 2,
            self.sep_token: 3,
        }

        self.id_to_token: Dict[int, str] = {
            0: self.pad_token,
            1: self.unk_token,
            2: self.cls_token,
            3: self.sep_token,
        }

        self.is_vocab_built = False

    def build_vocab(self, smiles_list: List[str]):
        """
        从训练集的 SMILES 列表中统计字符词表
        """
        unique_chars = set()

        for smiles in smiles_list:
            if smiles is None:
                continue
            smiles = str(smiles).strip()
            for ch in smiles:
                unique_chars.add(ch)

        # 排序后依次加入词表，保证结果稳定可复现
        sorted_chars = sorted(unique_chars)

        next_id = len(self.token_to_id)
        for ch in sorted_chars:
            if ch not in self.token_to_id:
                self.token_to_id[ch] = next_id
                self.id_to_token[next_id] = ch
                next_id += 1

        self.is_vocab_built = True

    def encode(self, smiles: str, max_length: int = 128):
        """
        将单条 SMILES 编码为固定长度的 input_ids 和 attention_mask
        """
        if not self.is_vocab_built:
            raise ValueError("词表尚未构建，请先调用 build_vocab()。")

        smiles = str(smiles).strip()

        # 字符切分
        tokens = list(smiles)

        # 加特殊 token
        tokens = [self.cls_token] + tokens + [self.sep_token]

        # token -> id
        input_ids = [
            self.token_to_id.get(token, self.token_to_id[self.unk_token])
            for token in tokens
        ]

        # 截断
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            input_ids[-1] = self.token_to_id[self.sep_token]

        # attention mask: 非 PAD 为 1
        attention_mask = [1] * len(input_ids)

        # padding
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.token_to_id[self.pad_token]] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def decode(self, input_ids: List[int]) -> str:
        """
        将 id 序列还原成 token 序列字符串，便于调试
        """
        tokens = [self.id_to_token.get(i, self.unk_token) for i in input_ids]
        return " ".join(tokens)

    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def save_vocab(self, path: str):
        """
        保存词表到 json 文件
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        """
        从 json 文件加载词表
        """
        with open(path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)

        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.is_vocab_built = True
