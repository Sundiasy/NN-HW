from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from rl_project.baselines.grpo import PolicyNet
from rl_project.baselines.ppo import ActorCritic
from rl_project.baselines.q_learning import discretize
from rl_project.sb3_trainers import SB3_ALGORITHMS


BASELINE_ALGORITHMS = ("baseline_q_learning", "baseline_ppo", "baseline_grpo")
ALGORITHMS = (*SB3_ALGORITHMS, *BASELINE_ALGORITHMS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained LunarLander-v3 agent.")
    parser.add_argument("--algo", choices=ALGORITHMS, required=True)
    parser.add_argument("--env-id", default="LunarLander-v3")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sleep", type=float, default=0.0, help="Extra delay per step in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make(args.env_id, render_mode="human")
    policy = load_policy(args, env)
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            for _ in range(args.max_steps):
                action = policy(np.asarray(obs, dtype=np.float32))
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if args.sleep > 0:
                    time.sleep(args.sleep)
                if terminated or truncated:
                    break
            print(f"episode={episode + 1}, reward={total_reward:.3f}")
    finally:
        env.close()


def load_policy(args: argparse.Namespace, env: gym.Env) -> Callable[[np.ndarray], int]:
    model_path = args.model_path or default_model_path(args.algo)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if args.algo in SB3_ALGORITHMS:
        model = load_sb3_model(args.algo, model_path, args.device)

        def policy(obs: np.ndarray) -> int:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        return policy

    if args.algo == "baseline_q_learning":
        data = np.load(model_path)
        q_table = data["q_table"]

        def policy(obs: np.ndarray) -> int:
            return int(np.argmax(q_table[discretize(obs)]))

        return policy

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = resolve_device(args.device)
    if args.algo == "baseline_ppo":
        model = ActorCritic(obs_dim, action_dim).to(device)

        def policy(obs: np.ndarray) -> int:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
            return int(torch.argmax(logits, dim=-1).item())

    elif args.algo == "baseline_grpo":
        model = PolicyNet(obs_dim, action_dim).to(device)

        def policy(obs: np.ndarray) -> int:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(obs_t)
            return int(torch.argmax(logits, dim=-1).item())

    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return policy


def load_sb3_model(algo: str, model_path: Path, device: str):
    if algo in {"q_learning", "dqn"}:
        from stable_baselines3 import DQN

        return DQN.load(str(model_path), device=device)
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(model_path), device=device)
    if algo == "a2c":
        from stable_baselines3 import A2C

        return A2C.load(str(model_path), device=device)
    if algo == "trpo":
        from sb3_contrib import TRPO

        return TRPO.load(str(model_path), device=device)
    raise ValueError(f"Unsupported SB3 algorithm: {algo}")


def default_model_path(algo: str) -> Path:
    if algo in SB3_ALGORITHMS:
        return Path("models") / f"{algo}_sb3.zip"
    if algo == "baseline_q_learning":
        return Path("models") / "baseline_q_learning.npz"
    return Path("models") / f"{algo}.pt"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


if __name__ == "__main__":
    main()
