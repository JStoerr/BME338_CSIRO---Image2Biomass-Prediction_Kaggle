"""Hyperparameter tuning for the MLP head using Ray Tune."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict

import torch
from ray import tune

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mlp_head  # noqa: E402


def trainable(config: Dict) -> None:
    """Ray Tune trainable that dispatches into the existing CLI logic."""
    args = mlp_head.build_argparser().parse_args([])
    args.hidden_dims = tuple(config["hidden_dims"])
    args.lr = config["lr"]
    args.dropout = config["dropout"]
    args.batch_size = config["batch_size"]
    args.epochs = config["epochs"]
    args.seed = config["seed"]
    args.embeddings = Path(config.get("embeddings", REPO_ROOT / "data/derived/dino_embeddings.parquet"))
    args.labels = Path(config.get("labels", REPO_ROOT / "data/train.csv"))

    ctx = tune.get_context()
    trial_name = ctx.get_trial_name() if ctx and ctx.get_trial_name() else f"trial_{config['seed']}"
    trial_dir = Path(config["base_output_dir"]) / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = trial_dir
    args.cpu = not torch.cuda.is_available()

    mlp_head.run(args)

    metrics_path = trial_dir / "mlp_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        report_dict = {}
        if "best_val_loss" in metrics:
            report_dict["best_val_loss"] = metrics.get("best_val_loss", float("inf"))
        if "val_r2" in metrics:
            report_dict["best_val_r2"] = metrics.get("val_r2")
        if report_dict:
            tune.report(metrics=report_dict)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ray Tune sweep for the MLP head.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Total Ray Tune samples to draw from the search space (defaults to 30 for a longer sweep).",
    )
    parser.add_argument(
        "--time-budget-s",
        type=int,
        default=None,
        help="Optional wall-clock budget for Ray Tune (seconds). Leave unset to disable.",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_argparser().parse_args()
    base_output_dir = REPO_ROOT / "data/derived" / "tune_runs"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = "mlp_head_tuning"
    search_space = {
        "hidden_dims": tune.grid_search([[128, 64],[256, 128], [512, 256], [256, 128, 64], [64, 32, 16, 16]]),
        "lr": tune.loguniform(1e-4, 1e-3),
        "dropout": tune.uniform(0.1, 0.3),
        "batch_size": tune.choice([64, 128]),
        "epochs": 200,
        "seed": tune.randint(0, 1000),
        "base_output_dir": str(base_output_dir),
    }

    tune_kwargs = {
        "num_samples": cli_args.num_samples,
    }
    if cli_args.time_budget_s:
        tune_kwargs["time_budget_s"] = cli_args.time_budget_s

    analysis = tune.run(
        trainable,
        config=search_space,
        metric="best_val_r2",
        mode="max",
        name=experiment_name,
        **tune_kwargs,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    history_records = []
    best_record = None
    for trial in analysis.trials:
        last_result = trial.last_result or {}
        record = {
            "trial_id": trial.trial_id,
            "status": trial.status,
            "local_path": trial.local_path,
            "config": trial.config,
            "best_val_r2": last_result.get("best_val_r2"),
            "best_val_loss": last_result.get("best_val_loss"),
        }
        history_records.append(record)
        if record["best_val_r2"] is not None:
            best_r2 = None if best_record is None else best_record.get("best_val_r2")
            if best_record is None or best_r2 is None or record["best_val_r2"] > best_r2:
                best_record = record

    run_summary = {
        "timestamp_utc": timestamp,
        "experiment_name": experiment_name,
        "num_trials": len(history_records),
        "best_trial": best_record,
        "trials": history_records,
    }

    summary_path = base_output_dir / f"run_{timestamp}_history.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))

    history_log = base_output_dir / "tune_history.jsonl"
    with history_log.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(run_summary) + "\n")

    latest_path = base_output_dir / "latest_run_summary.json"
    latest_path.write_text(json.dumps(run_summary, indent=2))

    if best_record:
        best_path = base_output_dir / "best_trial_summary.json"
        best_path.write_text(json.dumps(best_record, indent=2))
