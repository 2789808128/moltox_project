import sys
import torch
from rdkit import Chem
import transformers

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print("rdkit ok")
print("transformers:", transformers.__version__)