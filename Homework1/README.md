# NN-HW1

本作业基于 UCI `Concrete Compressive Strength` 数据集，使用 `PyTorch` 搭建多层感知机（MLP）完成混凝土抗压强度回归任务。

作业的报告见 `作业报告.pdf`


## 文件说明

- `作业报告.pdf`：作业报告。
- `作业报告.ipynb`：作业实验实现流程（报告notebook版本）。
- `train_mlp.py`：主训练脚本。
- `dataset/Concrete_Data.xls`：原始数据集。
- `*outputs/metrics.json`：训练后的主要指标。
- `*outputs/correlation_rank.csv`：特征与目标值的相关性排序。
- `*outputs/loss_curve.png`：训练/验证损失曲线。
- `*outputs/prediction_scatter.png`：测试集真实值与预测值散点图。
- `*outputs/correlation_bar.png`：特征相关性条形图。

## 运行方式

在 `torch` 环境中：

直接查看`作业报告.ipynb`


或使用训练脚本

```bash
python train_mlp.py
```
