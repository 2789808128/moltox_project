import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.models.inference.base_predictor import BasePredictor
from src.models.tokenizer import SmilesTokenizer
from src.models.fusion_model import SmilesMorganFusionModel


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


class FusionPredictor(BasePredictor):
    def __init__(self, project_root: str | None = None):
        super().__init__(project_root=project_root)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_names = TASK_NAMES
        self.threshold = 0.5
        self.model_name = "fusion"

        self.train_csv = self.project_root / "data" / "processed" / "tox21_train.csv"
        self.checkpoint_path = self.project_root / "outputs" / "checkpoints" / "fusion" / "best_fusion_model.pt"

        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.max_length = self.checkpoint.get("max_length", 64)
        self.fp_n_bits = self.checkpoint.get("fp_n_bits", 2048)
        self.fp_radius = 2
        self.model_config = self.checkpoint.get(
            "model_config",
            {
                "d_model": 128,
                "fp_hidden_dim": 256,
                "fusion_hidden_dim": 128,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.1,
                "num_tasks": 12,
                "pad_token_id": 0,
            },
        )

        self.morgan_generator = GetMorganGenerator(
            radius=self.fp_radius,
            fpSize=self.fp_n_bits
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
        model = SmilesMorganFusionModel(
            vocab_size=self.tokenizer.vocab_size(),
            fp_dim=self.fp_n_bits,
            d_model=self.model_config["d_model"],
            fp_hidden_dim=self.model_config["fp_hidden_dim"],
            fusion_hidden_dim=self.model_config["fusion_hidden_dim"],
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

    def _smiles_to_fp(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        fp = self.morgan_generator.GetFingerprint(mol)
        arr = np.zeros((self.fp_n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    @torch.no_grad()
    def predict(self, smiles: str):
        encoded = self.tokenizer.encode(smiles, max_length=self.max_length)
        fingerprint = self._smiles_to_fp(smiles)

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long).unsqueeze(0).to(self.device)
        fingerprint = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fingerprint=fingerprint
        )
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
            "model_type": "fusion",
            "model_name": self.model_name,
            "task_names": self.task_names,
            "threshold": self.threshold,
            "input_type": "smiles_sequence + morgan_fingerprint",
            "max_length": self.max_length,
            "fp_n_bits": self.fp_n_bits,
            "checkpoint_path": str(self.checkpoint_path),
        }


if __name__ == "__main__":
    predictor = FusionPredictor()
    result = predictor.predict("CCO")
    print(result)