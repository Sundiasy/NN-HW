# LunarLander-v3 Reinforcement Learning

本项目针对 `Gymnasium` 的 `LunarLander-v3` 环境，使用成熟强化学习库 `Stable-Baselines3` / `SB3-Contrib` 训练智能体：

- `q_learning`：使用 Stable-Baselines3 的 `DQN`，即 Q-Learning 的深度函数逼近版本
- `dqn`：同上，保留显式 DQN 名称
- `ppo`：使用 Stable-Baselines3 的 `PPO`
- `a2c`：使用 Stable-Baselines3 的 `A2C`
- `trpo`：使用 SB3-Contrib 的 `TRPO`

同时保留三种手写教学 baseline 用于对比：

- `baseline_q_learning`：离散化状态后的表格 Q-Learning
- `baseline_ppo`：手写 PPO
- `baseline_grpo`：手写 GRPO 风格组内相对优势优化

说明：`GRPO` 常见现成实现主要面向语言模型 RL，例如 TRL 的 `GRPOTrainer`，不适用于 `LunarLander-v3` 这类 Gymnasium 控制环境。因此 `baseline_grpo` 明确作为手写对照组，不作为成熟库算法。

评价指标覆盖 `要求.md` 中的三项：

- 平均奖励值：训练期平均奖励、最终滚动平均奖励、评估平均奖励
- 收敛速度：首次达到指定滚动平均奖励阈值的 episode
- 稳定性：最后若干 episode 奖励标准差

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

如果你已经单独安装了 CUDA 版 PyTorch，保持 `requirements.txt` 里的 `torch` 注释即可，避免 pip 覆盖 GPU 版 PyTorch。

如果 `gymnasium[box2d]` 安装失败，通常是缺少 Box2D/SWIG 构建环境；建议使用 Python 3.11 或 3.12 创建虚拟环境后重新安装。

## 快速运行

```bash
python train.py --algo q_learning --episodes 50 --eval-episodes 5
python train.py --algo ppo --episodes 50 --eval-episodes 5
python train.py --algo trpo --episodes 50 --eval-episodes 5
python train.py --algo baseline_grpo --episodes 50 --eval-episodes 5
```

统一运行库算法和手写 baseline：

```bash
python train_all.py --episodes 50 --eval-episodes 5
```

上传到 Weights & Biases：

```bash
wandb login
python train_all.py --episodes 50 --eval-episodes 5 --device auto --wandb --wandb-project lunarlander-rl
```

也可以只训练并上传单个算法或 baseline：

```bash
python train.py --algo ppo --episodes 300 --eval-episodes 10 --device auto --wandb --wandb-project lunarlander-rl
python train.py --algo baseline_ppo --episodes 300 --eval-episodes 10 --device auto --wandb --wandb-project lunarlander-rl
```

底层库按 `total_timesteps` 训练；如果没有显式传入，默认使用：

```text
total_timesteps = episodes * max_steps
```

也可以直接指定训练步数：

```bash
python train.py --algo ppo --total-timesteps 200000 --eval-episodes 20 --device auto
```

## 查看火箭落地 GUI

训练完成后可以用 `play.py` 打开 Gymnasium 的窗口回放策略：

```bash
python play.py --algo ppo --episodes 3 --device auto
python play.py --algo q_learning --episodes 3 --device auto
python play.py --algo baseline_grpo --episodes 3 --device auto
```

如果模型路径不是默认位置，可手动指定：

```bash
python play.py --algo ppo --model-path models/ppo_sb3.zip --episodes 3
```

输出文件默认写入 `results/`：

- `results/<algo>_train.csv`：每个 episode 的训练奖励
- `results/<algo>_metrics.json`：指标汇总
- `results/summary.csv`：多算法指标对比
- `results/learning_curves.png`：训练曲线
- `models/<algo>_sb3.zip`：Stable-Baselines3 模型
- `models/baseline_*.pt` 或 `models/baseline_q_learning.npz`：手写 baseline 模型

## 正式实验建议

为了得到更接近最优策略的结果，建议：

```bash
python train_all.py --algos q_learning ppo trpo baseline_q_learning baseline_ppo baseline_grpo --total-timesteps 500000 --episodes 1000 --eval-episodes 20 --convergence-threshold 200 --device auto
```

注意：库算法使用 `total_timesteps` 控制训练规模；手写 baseline 使用 `episodes` 控制训练规模。

其中 `LunarLander-v3` 通常以平均奖励达到 `200` 作为解决环境的参考阈值。
