from __future__ import annotations

from rl_project.baselines.grpo import train_baseline_grpo
from rl_project.baselines.ppo import train_baseline_ppo
from rl_project.baselines.q_learning import train_baseline_q_learning


BASELINE_ALGORITHMS = ("baseline_q_learning", "baseline_ppo", "baseline_grpo")

BASELINE_TRAINERS = {
    "baseline_q_learning": train_baseline_q_learning,
    "baseline_ppo": train_baseline_ppo,
    "baseline_grpo": train_baseline_grpo,
}

