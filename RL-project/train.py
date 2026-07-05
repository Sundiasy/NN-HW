from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rl_project.baseline_trainers import BASELINE_ALGORITHMS, BASELINE_TRAINERS
from rl_project.metrics import compute_metrics, save_learning_curve
from rl_project.sb3_trainers import SB3_ALGORITHMS, train_library_agent
from rl_project.wandb_utils import log_algorithm_result, log_summary_and_artifacts, start_wandb_run


ALGORITHMS = (*SB3_ALGORITHMS, *BASELINE_ALGORITHMS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agents on LunarLander-v3.")
    parser.add_argument("--algo", choices=ALGORITHMS, required=True)
    parser.add_argument("--env-id", default="LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--convergence-threshold", type=float, default=200.0)
    parser.add_argument("--rolling-window", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--sb3-verbose", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Upload metrics and artifacts to Weights & Biases.")
    parser.add_argument("--wandb-project", default="lunarlander-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    run = None
    if args.wandb:
        wandb, run = start_wandb_run(args)

    try:
        trainer = BASELINE_TRAINERS.get(args.algo, train_library_agent)
        result = trainer(args)
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise
    train_csv = args.output_dir / f"{args.algo}_train.csv"
    result.history.to_csv(train_csv, index=False)

    metrics = compute_metrics(
        result.history["reward"].tolist(),
        eval_rewards=result.eval_rewards,
        rolling_window=args.rolling_window,
        convergence_threshold=args.convergence_threshold,
    )
    metrics.update(
        {
            "algorithm": args.algo,
            "env_id": args.env_id,
            "episodes": args.episodes,
            "total_timesteps": args.total_timesteps or args.episodes * args.max_steps,
            "eval_episodes": args.eval_episodes,
            "model_path": str(result.model_path),
            "train_csv": str(train_csv),
        }
    )
    metrics_path = args.output_dir / f"{args.algo}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    curve_path = args.output_dir / "learning_curves.png"
    save_learning_curve({args.algo: result.history}, curve_path)
    if args.wandb:
        try:
            log_algorithm_result(wandb, args.algo, result.history, metrics, args.rolling_window)
            summary = pd.DataFrame([metrics])
            log_summary_and_artifacts(wandb, summary, curve_path, args.output_dir, args.model_dir)
        finally:
            run.finish()

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
