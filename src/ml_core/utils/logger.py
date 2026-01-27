import mlflow
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def start_run(config: dict, run_name: str = "default-run"):
    """Start an MLflow run and log config and seed."""
    mlflow.set_experiment(config.get("experiment_name", "tcga-fusion"))
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(config)

    seed = config.get("seed", 42)
    set_seed(seed)
    mlflow.log_param("seed", seed)


def log_metrics(metrics: dict, step: int = None):
    """Log metrics to MLflow."""
    for key, val in metrics.items():
        mlflow.log_metric(key, val, step=step)
