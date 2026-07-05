from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rl_project.metrics import rolling_mean


def require_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WandB is not installed. Install it with: python -m pip install wandb"
        ) from exc
    if not hasattr(wandb, "init"):
        location = getattr(wandb, "__file__", None) or getattr(wandb, "__path__", None)
        raise RuntimeError(
            "Imported an incomplete wandb module instead of the real package. "
            f"Imported location: {location}. Install/repair with: python -m pip install -U wandb"
        )
    return wandb


def wandb_config_from_args(args: Any) -> dict[str, Any]:
    return {
        "env_id": args.env_id,
        "episodes": args.episodes,
        "total_timesteps": args.total_timesteps or args.episodes * args.max_steps,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "rolling_window": args.rolling_window,
        "convergence_threshold": args.convergence_threshold,
        "max_steps": args.max_steps,
        "n_envs": args.n_envs,
        "device": args.device,
    }


def start_wandb_run(args: Any, algorithms: list[str] | None = None):
    wandb = require_wandb()
    if algorithms is None:
        run_name = args.wandb_run_name or f"{args.algo}-{args.env_id}-seed{args.seed}"
        config = {**wandb_config_from_args(args), "algorithm": args.algo}
    else:
        run_name = args.wandb_run_name or f"all-{args.env_id}-seed{args.seed}"
        config = {**wandb_config_from_args(args), "algorithms": algorithms}

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config=config,
    )
    url = run.get_url()
    if url:
        print(f"WandB run: {url}")
    return wandb, run


def log_algorithm_result(
    wandb: Any,
    algo: str,
    history: pd.DataFrame,
    metrics: dict[str, Any],
    rolling_window: int,
) -> None:
    rewards = history["reward"].tolist()
    rolling = rolling_mean(rewards, rolling_window)
    for idx, row in history.iterrows():
        wandb.log(
            {
                f"{algo}/reward": float(row["reward"]),
                f"{algo}/rolling_mean_reward": float(rolling[idx]),
                f"{algo}/steps": int(row["steps"]),
                f"{algo}/episode": int(row["episode"]),
                "latest_episode": int(row["episode"]),
            }
        )
    wandb.log(
        {
            f"final/{algo}/{key}": value
            for key, value in metrics.items()
            if _is_scalar(value)
        }
    )


def log_summary_and_artifacts(
    wandb: Any,
    summary: pd.DataFrame,
    curve_path: Path,
    output_dir: Path,
    model_dir: Path,
) -> None:
    wandb.log({"summary": wandb.Table(dataframe=summary)})
    _log_artifacts(wandb, curve_path, output_dir, model_dir)


def log_single_run(args: Any, history: pd.DataFrame, metrics: dict[str, Any], curve_path: Path) -> None:
    wandb = require_wandb()
    run_name = args.wandb_run_name or f"{args.algo}-{args.env_id}-seed{args.seed}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config={**wandb_config_from_args(args), "algorithm": args.algo},
    )
    try:
        rewards = history["reward"].tolist()
        rolling = rolling_mean(rewards, args.rolling_window)
        for idx, row in history.iterrows():
            wandb.log(
                {
                    "train/reward": float(row["reward"]),
                    "train/rolling_mean_reward": float(rolling[idx]),
                    "train/steps": int(row["steps"]),
                    "episode": int(row["episode"]),
                },
                step=int(row["episode"]),
            )
        wandb.log({f"final/{key}": value for key, value in metrics.items() if _is_scalar(value)})
        _log_artifacts(wandb, curve_path, args.output_dir, args.model_dir)
    finally:
        run.finish()


def log_all_run(
    args: Any,
    histories: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    curve_path: Path,
) -> None:
    wandb = require_wandb()
    run_name = args.wandb_run_name or f"all-{args.env_id}-seed{args.seed}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config={**wandb_config_from_args(args), "algorithms": list(histories.keys())},
    )
    try:
        for algo, history in histories.items():
            rewards = history["reward"].tolist()
            rolling = rolling_mean(rewards, args.rolling_window)
            for idx, row in history.iterrows():
                wandb.log(
                    {
                        f"{algo}/reward": float(row["reward"]),
                        f"{algo}/rolling_mean_reward": float(rolling[idx]),
                        f"{algo}/steps": int(row["steps"]),
                    },
                    step=int(row["episode"]),
                )
        wandb.log({"summary": wandb.Table(dataframe=summary)})
        for _, row in summary.iterrows():
            algo = row["algorithm"]
            wandb.log(
                {
                    f"final/{algo}/train_mean_reward": row["train_mean_reward"],
                    f"final/{algo}/eval_mean_reward": row["eval_mean_reward"],
                    f"final/{algo}/stability_last_window_std": row["stability_last_window_std"],
                    f"final/{algo}/best_rolling_mean_reward": row["best_rolling_mean_reward"],
                }
            )
        _log_artifacts(wandb, curve_path, args.output_dir, args.model_dir)
    finally:
        run.finish()


def _log_artifacts(wandb: Any, curve_path: Path, output_dir: Path, model_dir: Path) -> None:
    if curve_path.exists():
        wandb.log({"learning_curves": wandb.Image(str(curve_path))})

    artifact = wandb.Artifact("lunarlander-results", type="result")
    for directory in (output_dir, model_dir):
        if directory.exists():
            artifact.add_dir(str(directory))
    wandb.log_artifact(artifact)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (int, float, str, bool))
