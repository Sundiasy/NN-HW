from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from rl_project.algorithms.common import TrainResult


SB3_ALGORITHMS = ("q_learning", "dqn", "ppo", "a2c", "trpo")


def train_library_agent(args: Any) -> TrainResult:
    try:
        from stable_baselines3 import A2C, DQN, PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.monitor import load_results
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is required. Install with: "
            "python -m pip install stable-baselines3 sb3-contrib"
        ) from exc

    algo = args.algo
    if algo == "grpo":
        raise RuntimeError(
            "GRPO is not available as a Stable-Baselines3 Gymnasium algorithm. "
            "Use --algo ppo, --algo q_learning, --algo dqn, or --algo trpo."
        )

    model_cls, library_name = _resolve_algorithm(algo, DQN, PPO, A2C)
    total_timesteps = args.total_timesteps or args.episodes * args.max_steps
    n_envs = args.n_envs
    if library_name == "DQN" and n_envs != 1:
        n_envs = 1

    monitor_dir = Path(args.output_dir) / f"{algo}_monitor"
    if monitor_dir.exists():
        shutil.rmtree(monitor_dir)
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(args.env_id, n_envs=n_envs, seed=args.seed, monitor_dir=str(monitor_dir))
    kwargs = _default_model_kwargs(library_name, args)
    model = model_cls("MlpPolicy", env, seed=args.seed, verbose=args.sb3_verbose, **kwargs)
    model.learn(total_timesteps=total_timesteps, log_interval=args.log_interval, progress_bar=args.progress_bar)

    monitor_df = load_results(str(monitor_dir))
    history = _history_from_monitor(monitor_df)
    eval_env = gym.make(args.env_id)
    eval_rewards, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )
    eval_env.close()
    env.close()

    model_path = Path(args.model_dir) / f"{algo}_sb3.zip"
    model.save(str(model_path))
    return TrainResult(
        history=history,
        eval_rewards=[float(reward) for reward in eval_rewards],
        model_path=model_path,
    )


def _resolve_algorithm(algo: str, dqn_cls: Any, ppo_cls: Any, a2c_cls: Any) -> tuple[Any, str]:
    if algo in {"q_learning", "dqn"}:
        return dqn_cls, "DQN"
    if algo == "ppo":
        return ppo_cls, "PPO"
    if algo == "a2c":
        return a2c_cls, "A2C"
    if algo == "trpo":
        try:
            from sb3_contrib import TRPO
        except ImportError as exc:
            raise RuntimeError(
                "TRPO requires sb3-contrib. Install with: python -m pip install sb3-contrib"
            ) from exc
        return TRPO, "TRPO"
    raise ValueError(f"Unsupported algorithm: {algo}")


def _default_model_kwargs(library_name: str, args: Any) -> dict[str, Any]:
    device = args.device
    if library_name == "DQN":
        return {
            "learning_rate": 1e-3,
            "buffer_size": 100_000,
            "learning_starts": min(1_000, max(100, args.total_timesteps or args.episodes * args.max_steps // 10)),
            "batch_size": 64,
            "gamma": 0.99,
            "train_freq": 4,
            "target_update_interval": 1_000,
            "exploration_fraction": 0.25,
            "exploration_final_eps": 0.05,
            "device": device,
        }
    if library_name == "PPO":
        return {
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "device": device,
        }
    if library_name == "TRPO":
        return {
            "learning_rate": 1e-3,
            "n_steps": 1024,
            "batch_size": 128,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "device": device,
        }
    if library_name == "A2C":
        return {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "device": device,
        }
    return {"device": device}


def _history_from_monitor(monitor_df: pd.DataFrame) -> pd.DataFrame:
    if monitor_df.empty:
        return pd.DataFrame(columns=["episode", "reward", "steps"])
    rewards = monitor_df["r"].astype(float).to_numpy()
    lengths = monitor_df["l"].astype(int).to_numpy()
    return pd.DataFrame(
        {
            "episode": np.arange(1, len(rewards) + 1, dtype=int),
            "reward": rewards,
            "steps": lengths,
        }
    )
