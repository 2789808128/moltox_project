import os
import torch

from src.models.tokenizer import SmilesTokenizer
from src.models.transformer_model import SmilesTransformer


TASK_NAMES = [
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


class TransformerPredictor:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_csv = os.path.join(project_root, "data", "processed", "tox21_train.csv")
        self.checkpoint_path = os.path.join(
            project_root,
            "outputs",
            "checkpoints",
            "transformer",
            "best_smiles_transformer.pt"
        )

        self.max_length = 64
        self.threshold = 0.5

        self.tokenizer = self._build_tokenizer()
        self.model = self._load_model()

    def _build_tokenizer(self):
        import pandas as pd

        df = pd.read_csv(self.train_csv)
        train_smiles = df["smiles"].tolist()

        tokenizer = SmilesTokenizer()
        tokenizer.build_vocab(train_smiles)
        return tokenizer

    def _load_model(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        model = SmilesTransformer(
            vocab_size=self.tokenizer.vocab_size(),
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            max_len=self.max_length,
            num_tasks=12,
            pad_token_id=0,
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, smiles: str):
        encoded = self.tokenizer.encode(smiles, max_length=self.max_length)

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long).unsqueeze(0).to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        task_probs = {}
        task_preds = {}

        for task_name, prob in zip(TASK_NAMES, probs):
            prob = float(prob)
            pred = 1 if prob >= self.threshold else 0

            task_probs[task_name] = prob
            task_preds[task_name] = pred

        return {
            "model_name": "transformer",
            "smiles": smiles,
            "task_probs": task_probs,
            "task_preds": task_preds,
        }


if __name__ == "__main__":
    project_root = r"E:\Project\moltox_project"

    predictor = TransformerPredictor(project_root=project_root)

    smiles = "CCO"
    result = predictor.predict(smiles)

    print("Model:", result["model_name"])
    print("SMILES:", result["smiles"])
    print("\nTask probabilities:")
    for k, v in result["task_probs"].items():
        print(f"{k}: {v:.4f}")

    print("\nTask predictions:")
    for k, v in result["task_preds"].items():
        print(f"{k}: {v}")