# NN-HW1

本作业基于 UCI `Concrete Compressive Strength` 数据集，使用 `PyTorch` 搭建多层感知机（MLP）完成混凝土抗压强度回归任务。

## 任务完成情况

- 按题目要求使用前 `80%` 样本作为训练集、后 `20%` 样本作为测试集。
- 对训练集做特征相关性分析，并导出相关性排序与图表。
- 对输入特征做标准化处理。
- 使用 `torch` 实现 MLP 回归模型，并用验证集早停训练。
- 在测试集上计算 `MSE`，并绘制预测结果可视化图。

## 文件说明

- `train_mlp.py`：主训练脚本。
- `dataset/Concrete_Data.xls`：原始数据集。
- `outputs/metrics.json`：训练后的主要指标。
- `outputs/correlation_rank.csv`：特征与目标值的相关性排序。
- `outputs/loss_curve.png`：训练/验证损失曲线。
- `outputs/prediction_scatter.png`：测试集真实值与预测值散点图。
- `outputs/correlation_bar.png`：特征相关性条形图。
- `report.md`：作业报告摘要。

## 运行方式

在 `conda torch` 环境中执行：

```bash
python train_mlp.py
```

## 本次运行结果

- MLP 测试集 `MSE = 76.9802`
- MLP 测试集 `R^2 = 0.4957`
- 线性回归基线 `MSE = 75.7221`
- 线性回归基线 `R^2 = 0.5039`

说明：

- 在训练集上的相关性分析显示，`Cement`、`Superplasticizer`、`Age` 与目标值相关性最高。
- `Fly Ash` 与目标值的线性相关性最弱，但在当前固定切分下，保留全部 8 个特征的 MLP 表现更稳定。
- 为提高固定顺序切分场景下的训练稳定性，脚本对目标值使用了 `log1p` 变换后再训练。

## 依赖

```bash
pip install -r requirements.txt
```

实际训练结果与分析见 `outputs/` 和 `report.md`。
