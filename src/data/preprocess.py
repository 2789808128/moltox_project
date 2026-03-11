import os
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split


def is_valid_smiles(smiles: str) -> bool:
    """判断一个 SMILES 是否可以被 RDKit 正确解析"""
    if pd.isna(smiles):
        return False
    smiles = str(smiles).strip()
    if smiles == "":
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def print_label_statistics(df: pd.DataFrame, label_cols: list):
    """打印每个标签列的缺失值、0/1分布情况"""
    print("\n" + "=" * 60)
    print("标签统计信息")
    print("=" * 60)

    for col in label_cols:
        total = len(df)
        missing = df[col].isna().sum()
        valid = total - missing

        value_counts = df[col].value_counts(dropna=True).to_dict()
        num_0 = value_counts.get(0, 0)
        num_1 = value_counts.get(1, 0)

        print(f"\n任务: {col}")
        print(f"  总样本数: {total}")
        print(f"  有效标签数: {valid}")
        print(f"  缺失标签数: {missing}")
        print(f"  标签0数量: {num_0}")
        print(f"  标签1数量: {num_1}")


def split_data(df: pd.DataFrame, random_state: int = 42):
    """
    先划分 train 和 temp，再将 temp 划分为 valid 和 test
    最终比例约为 8:1:1
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        shuffle=True
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
        shuffle=True
    )

    return train_df, valid_df, test_df


def main():
    # 项目根目录
    project_root = r"E:\Project\moltox_project"

    # 输入输出路径
    input_path = os.path.join(project_root, "data", "raw", "tox21.csv")
    processed_dir = os.path.join(project_root, "data", "processed")

    clean_path = os.path.join(processed_dir, "tox21_clean.csv")
    train_path = os.path.join(processed_dir, "tox21_train.csv")
    valid_path = os.path.join(processed_dir, "tox21_valid.csv")
    test_path = os.path.join(processed_dir, "tox21_test.csv")

    os.makedirs(processed_dir, exist_ok=True)

    print("=" * 60)
    print("开始读取 Tox21 数据...")
    print(f"输入文件: {input_path}")

    # 读取原始数据
    df = pd.read_csv(input_path)

    print("\n原始数据基本信息：")
    print(f"行数: {df.shape[0]}")
    print(f"列数: {df.shape[1]}")
    print("\n列名如下：")
    print(df.columns.tolist())

    # 检测 SMILES 列
    possible_smiles_cols = ["smiles", "SMILES", "mol", "molecule"]
    smiles_col = None
    for col in possible_smiles_cols:
        if col in df.columns:
            smiles_col = col
            break

    if smiles_col is None:
        raise ValueError("没有找到 SMILES 列，请检查原始 csv 的列名。")

    print(f"\n检测到 SMILES 列: {smiles_col}")

    # 明确标签列
    non_label_cols = {smiles_col, "mol_id", "id", "ID", "sample_id"}
    label_cols = [col for col in df.columns if col not in non_label_cols]

    print(f"\n识别到标签列数量: {len(label_cols)}")
    print("标签列名称：")
    print(label_cols)

    # 删除空 SMILES
    before_dropna = len(df)
    df = df[df[smiles_col].notna()].copy()
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col] != ""].copy()
    after_dropna = len(df)

    print("\n删除空 SMILES 后：")
    print(f"剩余样本数: {after_dropna}")
    print(f"删除样本数: {before_dropna - after_dropna}")

    # 合法性检查
    print("\n开始检查 SMILES 合法性...")
    df["is_valid"] = df[smiles_col].apply(is_valid_smiles)

    valid_count = df["is_valid"].sum()
    invalid_count = len(df) - valid_count

    print(f"合法 SMILES 数量: {valid_count}")
    print(f"非法 SMILES 数量: {invalid_count}")

    # 保留合法数据
    df_clean = df[df["is_valid"]].copy()
    df_clean.drop(columns=["is_valid"], inplace=True)

    print("\n清洗后数据基本信息：")
    print(f"行数: {df_clean.shape[0]}")
    print(f"列数: {df_clean.shape[1]}")

    # 打印标签统计
    print_label_statistics(df_clean, label_cols)

    # 保存 clean 数据
    df_clean.to_csv(clean_path, index=False)
    print(f"\n已保存清洗后数据: {clean_path}")

    # 数据集划分
    train_df, valid_df, test_df = split_data(df_clean, random_state=42)

    print("\n" + "=" * 60)
    print("数据集划分结果")
    print("=" * 60)
    print(f"训练集数量: {len(train_df)}")
    print(f"验证集数量: {len(valid_df)}")
    print(f"测试集数量: {len(test_df)}")
    print(f"总计: {len(train_df) + len(valid_df) + len(test_df)}")

    # 保存划分结果
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n已保存数据划分文件：")
    print(train_path)
    print(valid_path)
    print(test_path)

    print("\n预处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()