import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"
TEST_CSV = r"E:\Project\moltox_project\data\processed\tox21_test.csv"

# 你可以改这里测试不同模型
MODEL_TYPES = ["transformer", "fusion", "morgan_logreg", "morgan_rf"]

df = pd.read_csv(TEST_CSV)

# 取前 200 条先试，够快
smiles_list = df["smiles"].dropna().tolist()[:200]

for model_type in MODEL_TYPES:
    print("\n" + "=" * 60)
    print("MODEL:", model_type)
    print("=" * 60)

    found = 0

    for smiles in smiles_list:
        payload = {
            "model_type": model_type,
            "smiles": smiles
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            positive_tasks = [
                task for task, pred in data["task_preds"].items()
                if pred == 1
            ]

            if positive_tasks:
                print("SMILES:", smiles)
                print("Positive tasks:", positive_tasks)
                print("Probabilities:")
                for task in positive_tasks:
                    print(f"  {task}: {data['task_probs'][task]:.4f}")
                print("-" * 40)

                found += 1

            if found >= 5:
                break

        except Exception as e:
            print("Request failed for:", smiles)
            print("Error:", e)

    if found == 0:
        print("前 200 条里没有找到预测为 1 的例子。")