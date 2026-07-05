from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from rl_project.envs import make_env, reset_env, step_env


@dataclass
class TrainResult:
    history: pd.DataFrame
    eval_rewards: list[float]
    model_path: Path


def evaluate_policy(
    env_id: str,
    policy: Callable[[np.ndarray], int],
    episodes: int,
    seed: int,
    max_steps: int,
) -> list[float]:
    env = make_env(env_id, seed=seed)
    rewards: list[float] = []
    for episode in range(episodes):
        obs = reset_env(env, seed=seed + 10_000 + episode)
        total = 0.0
        for _ in range(max_steps):
            obs, reward, done, _ = step_env(env, policy(obs))
            total += reward
            if done:
                break
        rewards.append(total)
    env.close()
    return rewards

