from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import trange

from rl_project.algorithms.common import TrainResult, evaluate_policy
from rl_project.envs import make_env, reset_env, set_global_seed, step_env


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        return self.actor(features), self.critic(features).squeeze(-1)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self(obs_t)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_baseline_ppo(args) -> TrainResult:
    set_global_seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    env = make_env(args.env_id, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    update_epochs = 4
    batch_size = 256
    entropy_coef = 0.01
    value_coef = 0.5

    rows: list[dict[str, float | int]] = []
    for episode in trange(args.episodes, desc="baseline_ppo", leave=False):
        obs_buf: list[np.ndarray] = []
        action_buf: list[int] = []
        logprob_buf: list[float] = []
        reward_buf: list[float] = []
        value_buf: list[float] = []
        done_buf: list[bool] = []

        obs = reset_env(env, seed=args.seed + episode)
        total = 0.0
        for step in range(args.max_steps):
            action, log_prob, value = model.act(obs)
            next_obs, reward, done, _ = step_env(env, action)
            obs_buf.append(obs)
            action_buf.append(action)
            logprob_buf.append(log_prob)
            reward_buf.append(reward)
            value_buf.append(value)
            done_buf.append(done)
            total += reward
            obs = next_obs
            if done:
                break

        advantages, returns = compute_gae(reward_buf, value_buf, done_buf, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(action_buf, dtype=torch.long, device=device)
        old_logprobs_t = torch.as_tensor(logprob_buf, dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        indices = np.arange(len(obs_buf))
        for _ in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                logits, values = model(obs_t[batch])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions_t[batch])
                ratio = torch.exp(log_probs - old_logprobs_t[batch])
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                policy_loss = -torch.min(ratio * advantages_t[batch], clipped * advantages_t[batch]).mean()
                value_loss = (returns_t[batch] - values).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        rows.append({"episode": episode + 1, "reward": total, "steps": step + 1})

    env.close()

    def policy(obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    eval_rewards = evaluate_policy(args.env_id, policy, args.eval_episodes, args.seed, args.max_steps)
    model_path = Path(args.model_dir) / "baseline_ppo.pt"
    torch.save(model.state_dict(), model_path)
    return TrainResult(pd.DataFrame(rows), eval_rewards, model_path)

