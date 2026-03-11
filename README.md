# moltox_project

基于 **Tox21 数据集** 的多任务分子毒性预测系统。  
输入为 **SMILES**，输出为 12 个毒性相关任务的概率和二分类结果。

本项目不是单纯的训练脚本，而是一个完整的小型系统，包含：

- 数据预处理
- 多模型训练与测试
- FastAPI 后端预测接口
- Vue3 前端可视化展示
- 多模型对比功能

---

## 1. 项目功能概览

当前系统支持 4 种模型：

1. **transformer**
   - 字符级 SMILES Transformer
2. **morgan_logreg**
   - Morgan Fingerprint + Logistic Regression
3. **morgan_rf**
   - Morgan Fingerprint + Random Forest
4. **fusion**
   - SMILES Transformer + Morgan Fingerprint 特征级融合模型

系统支持：

- 单模型预测
- 四模型对比预测
- 12 个毒性任务概率展示
- 风险摘要展示
- 前后端联动调用

---

## 2. 项目结构

```text
moltox_project/
├─ configs/                     # 模型配置文件
├─ data/
│  ├─ raw/                      # 原始数据
│  └─ processed/                # 清洗与划分后的数据
├─ frontend/                    # Vue3 前端
├─ outputs/
│  ├─ checkpoints/              # 模型权重与传统模型文件
│  ├─ logs/                     # 训练日志、history、曲线
│  └─ experiments/              # 实验记录（建议维护）
├─ src/
│  ├─ api/                      # FastAPI 接口层
│  ├─ data/                     # Dataset / Dataloader / preprocess
│  ├─ engine/                   # train / test / evaluate / loss / metrics
│  ├─ models/                   # 模型定义、tokenizer、inference
│  └─ utils/                    # 工具函数
└─ README.md
3. 开发环境

已验证环境：

Windows

Miniconda

PyCharm

Python 3.10.19

PyTorch 2.10.0+cu126

CUDA available

GPU: NVIDIA GeForce RTX 3050 Laptop GPU

RDKit

Transformers

推荐 conda 环境名称：

moltox
4. 数据集说明

使用数据集：Tox21

特点：

多任务毒性预测

共 12 个毒性任务

输入为 SMILES

标签存在缺失值

项目中采用：

RDKit 校验 SMILES 合法性

训练 / 验证 / 测试集划分

label_mask 处理缺失标签

多任务平均 ROC-AUC 作为主要评估指标

5. 已实现模型
5.1 Transformer

字符级 tokenizer

Embedding + PositionalEncoding + TransformerEncoder

masked mean pooling

多任务输出

5.2 Morgan + Logistic Regression

Morgan Fingerprint

每个任务训练一个 Logistic Regression 分类器

5.3 Morgan + Random Forest

Morgan Fingerprint

每个任务训练一个 Random Forest 分类器

5.4 Fusion

SMILES Transformer 分支

Morgan Fingerprint MLP 分支

特征级融合输出

6. 当前结果（第一轮优化后）

当前已知核心结果：

Transformer Test Mean ROC-AUC: 0.7554

Logistic Regression Mean ROC-AUC: 0.7858

Random Forest Mean ROC-AUC: 0.8131

Fusion Test Mean ROC-AUC: 0.8026

当前表现最强的模型是：

Morgan + Random Forest

Fusion 也有较强综合表现，并且更具系统与研究展示价值

7. 第一轮优化已完成内容

已完成以下升级：

训练脚本升级

train.py

train_fusion.py

新增：

更长训练轮数

early stopping

learning rate scheduler

grad clip

更完整 checkpoint 保存内容

测试脚本升级

test.py

test_fusion.py

实现：

从 checkpoint 自动恢复模型配置

传统模型管理升级

新增 train_ml_models.py

传统模型离线训练后保存为 joblib

predict_logreg.py / predict_rf.py 改为直接加载模型文件

predictor 接口统一

四个 predictor 已统一接口风格：

predict(smiles)

get_metadata()

并通过 ModelRouter 统一调度。

8. 运行方式
8.1 数据预处理

根据需要运行：

python src/data/preprocess.py
8.2 训练深度模型
Transformer
python src/engine/train.py
Fusion
python src/engine/train_fusion.py
8.3 测试深度模型
Transformer
python src/engine/test.py
Fusion
python src/engine/test_fusion.py
8.4 训练并保存传统模型
python src/models/train_ml_models.py

运行后会生成：

outputs/checkpoints/ml_baselines/morgan_logreg.joblib

outputs/checkpoints/ml_baselines/morgan_rf.joblib

8.5 单独测试 predictor
python src/models/inference/predict_transformer.py
python src/models/inference/predict_fusion.py
python src/models/inference/predict_logreg.py
python src/models/inference/predict_rf.py
8.6 启动后端
python src/api/service.py

默认预测接口：

POST /predict

请求体示例：

{
  "model_type": "fusion",
  "smiles": "CCO"
}
8.7 启动前端

进入前端目录后运行：

npm install
npm run dev
9. 前端页面说明

前端为 Vue3 多页面系统，包括：

HomeView：项目总览

KnowledgeView：方法介绍、分子指纹互动展示

PredictView：核心预测与多模型对比

SystemDesignView：系统设计展示

10. 后续优化方向

建议优先级：

路径与配置进一步规范化

实验记录标准化

Transformer / Fusion 第二轮性能优化

前端多模型差异展示增强

11. 项目特点总结

本项目已经完成：

数据、训练、测试、后端、前端全链路打通

多模型统一预测

多任务毒性预测系统化展示

第一轮工程与训练规范化优化

当前项目状态适合继续进行：

第二轮性能优化

答辩展示增强

实验对比深化
## 刷新前端实验结果图片

当训练、测试或结果图更新后，可以运行下面的命令自动重画并同步前端静态资源：

```bash
python src/utils/refresh_frontend_assets.py

该脚本会自动：

读取最新实验记录中的 test 结果

重画模型总体对比图

重画任务级对比图

同步最新的 Transformer / Fusion 训练曲线图

覆盖更新 frontend/public/ 下的前端展示图片
## 实验记录与前端结果图自动刷新

项目已支持将实验结果自动记录到：

```text
outputs/experiments/experiment_registry.csv

其中包括：

Transformer 测试结果

Fusion 测试结果

Morgan + Logistic Regression 测试结果

Morgan + Random Forest 测试结果

自动刷新前端展示图片

当测试结果或实验结果更新后，可以运行下面的命令自动刷新前端静态资源：

python src/utils/refresh_frontend_assets.py

该脚本会自动完成以下操作：

读取 outputs/experiments/experiment_registry.csv 中最新的 test 结果

重画模型总体对比图 model_comparison_mean_auc.png

重画任务级对比图 task_auc_comparison.png

同步最新的 Transformer / Fusion 训练曲线图

自动覆盖更新 frontend/public/ 下的前端展示图片

当前自动化流程说明

运行 src/engine/test.py 后，会：

输出 Transformer 测试结果

自动写入实验记录

自动刷新前端图片

运行 src/engine/test_fusion.py 后，会：

输出 Fusion 测试结果

自动写入实验记录

自动刷新前端图片

运行 src/models/train_ml_models.py 后，会：

训练并保存 morgan_logreg.joblib

训练并保存 morgan_rf.joblib

自动评估传统模型测试集结果

自动写入实验记录

推荐使用顺序

当你完成新一轮优化后，推荐按如下顺序操作：

python src/models/train_ml_models.py
python src/engine/test.py
python src/engine/test_fusion.py
python src/utils/refresh_frontend_assets.py

这样可以确保：

四个模型的实验结果已更新

实验登记表已更新

前端展示图片已同步到最新状态
