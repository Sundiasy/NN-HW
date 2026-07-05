from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import trange

from rl_project.algorithms.common import TrainResult, evaluate_policy
from rl_project.baselines.ppo import resolve_device
from rl_project.envs import make_env, reset_env, set_global_seed, step_env


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, float]:
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = self(obs_t)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())


def train_baseline_grpo(args) -> TrainResult:
    set_global_seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    env = make_env(args.env_id, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = PolicyNet(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-4)

    group_size = 4
    clip_eps = 0.2
    update_epochs = 4
    entropy_coef = 0.02
    rows: list[dict[str, float | int]] = []

    for group_start in trange(0, args.episodes, group_size, desc="baseline_grpo", leave=False):
        trajectories = []
        returns = []
        actual_group = min(group_size, args.episodes - group_start)
        for local_idx in range(actual_group):
            episode = group_start + local_idx
            obs_buf: list[np.ndarray] = []
            action_buf: list[int] = []
            logprob_buf: list[float] = []
            obs = reset_env(env, seed=args.seed + episode)
            total = 0.0
            for step in range(args.max_steps):
                action, log_prob = policy_net.act(obs)
                next_obs, reward, done, _ = step_env(env, action)
                obs_buf.append(obs)
                action_buf.append(action)
                logprob_buf.append(log_prob)
                total += reward
                obs = next_obs
                if done:
                    break
            trajectories.append((obs_buf, action_buf, logprob_buf, step + 1, total))
            returns.append(total)
            rows.append({"episode": episode + 1, "reward": total, "steps": step + 1})

        group_mean = float(np.mean(returns))
        group_std = float(np.std(returns) + 1e-8)
        obs_all: list[np.ndarray] = []
        action_all: list[int] = []
        old_logprob_all: list[float] = []
        advantage_all: list[float] = []
        for obs_buf, action_buf, logprob_buf, _, total in trajectories:
            relative_advantage = (total - group_mean) / group_std
            obs_all.extend(obs_buf)
            action_all.extend(action_buf)
            old_logprob_all.extend(logprob_buf)
            advantage_all.extend([relative_advantage] * len(obs_buf))

        obs_t = torch.as_tensor(np.asarray(obs_all), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(action_all, dtype=torch.long, device=device)
        old_logprobs_t = torch.as_tensor(old_logprob_all, dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantage_all, dtype=torch.float32, device=device)

        for _ in range(update_epochs):
            logits = policy_net(obs_t)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)
            ratio = torch.exp(log_probs - old_logprobs_t)
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = -torch.min(ratio * advantages_t, clipped * advantages_t).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
            optimizer.step()

    env.close()

    def policy(obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = policy_net(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    eval_rewards = evaluate_policy(args.env_id, policy, args.eval_episodes, args.seed, args.max_steps)
    model_path = Path(args.model_dir) / "baseline_grpo.pt"
    torch.save(policy_net.state_dict(), model_path)
    return TrainResult(pd.DataFrame(rows), eval_rewards, model_path)

