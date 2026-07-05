from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    obs, _ = env.reset(seed=seed)
    return np.asarray(obs, dtype=np.float32)


def step_env(env: gym.Env, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    obs, reward, terminated, truncated, info = env.step(action)
    return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated or truncated), info

