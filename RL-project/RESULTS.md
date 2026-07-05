# LunarLander-v3 实验结果

当前项目已改为使用成熟库实现：

- `q_learning` / `dqn`：Stable-Baselines3 `DQN`
- `ppo`：Stable-Baselines3 `PPO`
- `a2c`：Stable-Baselines3 `A2C`
- `trpo`：SB3-Contrib `TRPO`

同时保留手写 baseline 用于对比：

- `baseline_q_learning`：离散化状态后的表格 Q-Learning
- `baseline_ppo`：手写 PPO
- `baseline_grpo`：手写 GRPO 风格组内相对优势优化

旧结果已删除，需要在新环境安装依赖后重新训练生成。

## 重新训练

快速验证：

```bash
python train_all.py --algos q_learning ppo trpo baseline_q_learning baseline_ppo baseline_grpo --episodes 50 --total-timesteps 50000 --eval-episodes 5 --max-steps 500 --rolling-window 20 --convergence-threshold 200 --device auto
```

正式实验：

```bash
python train_all.py --algos q_learning ppo trpo baseline_q_learning baseline_ppo baseline_grpo --episodes 1000 --total-timesteps 500000 --eval-episodes 20 --rolling-window 50 --convergence-threshold 200 --device auto
```

上传到 WandB：

```bash
python train_all.py --algos q_learning ppo trpo baseline_q_learning baseline_ppo baseline_grpo --episodes 1000 --total-timesteps 500000 --eval-episodes 20 --rolling-window 50 --convergence-threshold 200 --device auto --wandb --wandb-project lunarlander-rl
```

## 指标说明

- 平均奖励值：`train_mean_reward` 表示训练期间平均奖励，`eval_mean_reward` 表示训练后评估平均奖励。
- 收敛速度：`convergence_episode` 表示首次达到指定滚动平均奖励阈值的 episode；为空表示当前训练规模内尚未达到。
- 稳定性：`stability_last_window_std` 表示最后若干训练 episode 的奖励标准差，越低代表长期表现越稳定。

## 输出文件

训练完成后会生成：

- `results/summary.csv`：多算法指标汇总
- `results/<algo>_metrics.json`：单算法指标
- `results/<algo>_train.csv`：训练奖励曲线数据
- `results/learning_curves.png`：学习曲线图
- `models/<algo>_sb3.zip`：Stable-Baselines3 模型
- `models/baseline_*.pt` 或 `models/baseline_q_learning.npz`：手写 baseline 模型

## 关于 GRPO

GRPO 的现成库实现主要面向语言模型 RL，不适用于 `LunarLander-v3` 这类 Gymnasium 控制环境。当前项目把 `baseline_grpo` 明确标为手写对照组，不把它当作成熟库算法；库算法对比建议优先使用 `q_learning`/`dqn`、`ppo`、`a2c`、`trpo`。
