from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from rl_project.baseline_trainers import BASELINE_ALGORITHMS, BASELINE_TRAINERS
from rl_project.metrics import compute_metrics, save_learning_curve
from rl_project.sb3_trainers import SB3_ALGORITHMS, train_library_agent
from rl_project.wandb_utils import log_algorithm_result, log_summary_and_artifacts, start_wandb_run


DEFAULT_ALGOS = ["q_learning", "ppo", "trpo", "baseline_q_learning", "baseline_ppo", "baseline_grpo"]
ALGORITHMS = (*SB3_ALGORITHMS, *BASELINE_ALGORITHMS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all LunarLander-v3 agents.")
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS, choices=ALGORITHMS)
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
    parser.add_argument("--device", default="auto")
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
    base_args = parse_args()
    base_args.output_dir.mkdir(parents=True, exist_ok=True)
    base_args.model_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    run = None
    if base_args.wandb:
        wandb, run = start_wandb_run(base_args, algorithms=list(base_args.algos))

    histories = {}
    rows = []
    try:
        for offset, algo in enumerate(base_args.algos):
            args = SimpleNamespace(**vars(base_args))
            args.algo = algo
            args.seed = base_args.seed + offset

            print(f"\n=== Training {algo} ===")
            trainer = BASELINE_TRAINERS.get(algo, train_library_agent)
            if algo in BASELINE_TRAINERS:
                result = trainer(args)
                train_time_seconds = None
            else:
                start_time = time.perf_counter()
                result = trainer(args)
                train_time_seconds = time.perf_counter() - start_time
            histories[algo] = result.history

            train_csv = args.output_dir / f"{algo}_train.csv"
            result.history.to_csv(train_csv, index=False)
            metrics = compute_metrics(
                result.history["reward"].tolist(),
                eval_rewards=result.eval_rewards,
                rolling_window=args.rolling_window,
                convergence_threshold=args.convergence_threshold,
            )
            metrics.update(
                {
                    "algorithm": algo,
                    "env_id": args.env_id,
                    "episodes": args.episodes,
                    "total_timesteps": args.total_timesteps or args.episodes * args.max_steps,
                    "eval_episodes": args.eval_episodes,
                    "model_path": str(result.model_path),
                    "train_csv": str(train_csv),
                }
            )
            if train_time_seconds is not None:
                metrics.update(
                    {
                        "train_time_seconds": round(train_time_seconds, 3),
                        "train_time_minutes": round(train_time_seconds / 60.0, 3),
                    }
                )
                print(f"{algo} elapsed time: {train_time_seconds:.2f}s")
            (args.output_dir / f"{algo}_metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            rows.append(metrics)
            if base_args.wandb:
                log_algorithm_result(wandb, algo, result.history, metrics, args.rolling_window)
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise

    if run is not None and not rows:
        run.finish(exit_code=1)
        raise RuntimeError("No training results were produced.")

    summary = pd.DataFrame(rows)
    summary_path = base_args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    curve_path = base_args.output_dir / "learning_curves.png"
    save_learning_curve(histories, curve_path)
    if base_args.wandb:
        try:
            log_summary_and_artifacts(wandb, summary, curve_path, base_args.output_dir, base_args.model_dir)
        finally:
            run.finish()
    print(f"\nSaved summary to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
