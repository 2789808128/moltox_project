import pandas as pd
import torch

from src.models.inference.base_predictor import BasePredictor
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


class TransformerPredictor(BasePredictor):
    def __init__(self, project_root: str | None = None):
        super().__init__(project_root=project_root)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_names = TASK_NAMES
        self.threshold = 0.5
        self.model_name = "transformer"

        self.train_csv = self.project_root / "data" / "processed" / "tox21_train.csv"
        self.checkpoint_path = (
            self.project_root / "outputs" / "checkpoints" / "transformer" / "best_smiles_transformer.pt"
        )

        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.max_length = self.checkpoint.get("max_length", 64)
        self.model_config = self.checkpoint.get(
            "model_config",
            {
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                "num_tasks": 12,
                "pad_token_id": 0,
            },
        )

        self.tokenizer = self._build_tokenizer()
        self.model = self._load_model()

    def _build_tokenizer(self):
        df = pd.read_csv(self.train_csv)
        smiles_col = "smiles" if "smiles" in df.columns else "SMILES"
        train_smiles = df[smiles_col].tolist()

        tokenizer = SmilesTokenizer()
        tokenizer.build_vocab(train_smiles)
        return tokenizer

    def _load_model(self):
        model = SmilesTransformer(
            vocab_size=self.tokenizer.vocab_size(),
            d_model=self.model_config["d_model"],
            nhead=self.model_config["nhead"],
            num_layers=self.model_config["num_layers"],
            dim_feedforward=self.model_config["dim_feedforward"],
            dropout=self.model_config["dropout"],
            max_len=self.max_length,
            num_tasks=self.model_config["num_tasks"],
            pad_token_id=self.model_config["pad_token_id"],
        ).to(self.device)

        model.load_state_dict(self.checkpoint["model_state_dict"])
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

        for task_name, prob in zip(self.task_names, probs):
            prob = float(prob)
            pred = 1 if prob >= self.threshold else 0
            task_probs[task_name] = prob
            task_preds[task_name] = pred

        return self.format_result(
            model_name=self.model_name,
            smiles=smiles,
            task_probs=task_probs,
            task_preds=task_preds,
        )

    def get_metadata(self):
        return {
            "model_type": "transformer",
            "model_name": self.model_name,
            "task_names": self.task_names,
            "threshold": self.threshold,
            "input_type": "smiles_sequence",
            "max_length": self.max_length,
            "checkpoint_path": str(self.checkpoint_path),
        }


if __name__ == "__main__":
    predictor = TransformerPredictor()
    result = predictor.predict("CCO")
    print(result)