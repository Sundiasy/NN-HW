from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)
    series = pd.Series(values, dtype="float32")
    return series.rolling(window=min(window, len(values)), min_periods=1).mean().to_numpy()


def compute_metrics(
    train_rewards: list[float],
    eval_rewards: list[float],
    rolling_window: int,
    convergence_threshold: float,
) -> dict[str, float | int | None]:
    rewards = np.asarray(train_rewards, dtype=np.float32)
    eval_arr = np.asarray(eval_rewards, dtype=np.float32)
    roll = rolling_mean(train_rewards, rolling_window)
    convergence_episode = None
    for idx, value in enumerate(roll, start=1):
        if value >= convergence_threshold:
            convergence_episode = idx
            break

    tail_count = min(rolling_window, len(rewards))
    tail = rewards[-tail_count:] if tail_count else np.array([], dtype=np.float32)
    return {
        "train_mean_reward": round(float(np.mean(rewards)), 3) if rewards.size else None,
        "final_rolling_mean_reward": round(float(roll[-1]), 3) if roll.size else None,
        "best_rolling_mean_reward": round(float(np.max(roll)), 3) if roll.size else None,
        "eval_mean_reward": round(float(np.mean(eval_arr)), 3) if eval_arr.size else None,
        "eval_std_reward": round(float(np.std(eval_arr)), 3) if eval_arr.size else None,
        "stability_last_window_std": round(float(np.std(tail)), 3) if tail.size else None,
        "convergence_episode": convergence_episode,
        "convergence_threshold": convergence_threshold,
        "rolling_window": rolling_window,
    }


def save_learning_curve(histories: dict[str, pd.DataFrame], path: Path) -> None:
    if not histories:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for algo, history in histories.items():
        rewards = history["reward"].tolist()
        roll = rolling_mean(rewards, min(50, max(1, len(rewards))))
        plt.plot(history["episode"], roll, label=f"{algo} rolling mean")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("LunarLander-v3 Training Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
