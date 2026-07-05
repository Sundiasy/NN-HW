from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

from rl_project.algorithms.common import TrainResult, evaluate_policy
from rl_project.envs import make_env, reset_env, set_global_seed, step_env


LOW = np.array([-1.5, -0.5, -2.0, -2.0, -3.1416, -5.0, 0.0, 0.0], dtype=np.float32)
HIGH = np.array([1.5, 1.5, 2.0, 2.0, 3.1416, 5.0, 1.0, 1.0], dtype=np.float32)
BINS = np.array([8, 8, 8, 8, 8, 8, 2, 2], dtype=np.int32)


def discretize(obs: np.ndarray) -> tuple[int, ...]:
    clipped = np.clip(obs, LOW, HIGH)
    scaled = (clipped - LOW) / (HIGH - LOW)
    indices = np.floor(scaled * BINS).astype(np.int32)
    indices = np.minimum(indices, BINS - 1)
    return tuple(indices.tolist())


def epsilon_by_episode(episode: int, episodes: int) -> float:
    progress = episode / max(1, episodes - 1)
    return max(0.05, 1.0 - progress * 0.95)


def train_baseline_q_learning(args) -> TrainResult:
    set_global_seed(args.seed)
    env = make_env(args.env_id, seed=args.seed)
    action_count = env.action_space.n
    q_table = np.zeros(tuple(BINS.tolist()) + (action_count,), dtype=np.float32)

    alpha = 0.12
    gamma = 0.99
    rows: list[dict[str, float | int]] = []
    for episode in trange(args.episodes, desc="baseline_q_learning", leave=False):
        obs = reset_env(env, seed=args.seed + episode)
        state = discretize(obs)
        total = 0.0
        epsilon = epsilon_by_episode(episode, args.episodes)

        for step in range(args.max_steps):
            if np.random.rand() < epsilon:
                action = int(env.action_space.sample())
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, done, _ = step_env(env, action)
            next_state = discretize(next_obs)
            target = reward + (0.0 if done else gamma * float(np.max(q_table[next_state])))
            q_table[state + (action,)] += alpha * (target - q_table[state + (action,)])
            state = next_state
            total += reward
            if done:
                break

        rows.append({"episode": episode + 1, "reward": total, "steps": step + 1, "epsilon": epsilon})

    env.close()

    def policy(obs: np.ndarray) -> int:
        return int(np.argmax(q_table[discretize(obs)]))

    eval_rewards = evaluate_policy(args.env_id, policy, args.eval_episodes, args.seed, args.max_steps)
    model_path = Path(args.model_dir) / "baseline_q_learning.npz"
    np.savez_compressed(model_path, q_table=q_table, bins=BINS, low=LOW, high=HIGH)
    return TrainResult(pd.DataFrame(rows), eval_rewards, model_path)

