from pathlib import Path
from typing import Dict, Union, Tuple, List, Callable, Optional


import wandb
from tqdm import tqdm

# Pandas
import pandas as pd

# RS-DDPM
from remote_sensing_ddpm.evaluation.data_handling.constants import (
    Resolution,
)


def get_wandb_run_histories(
    project_id: str,
    run_ids: List[str],
) -> Dict[str, Union[str, Dict[str, float]]]:
    api = wandb.Api(timeout=15)
    runs = api.runs(project_id)
    run_results = {}
    for run in [r for r in runs if r.id in run_ids]:
        single_run_complete_history = []
        # Get all data
        for x in tqdm(run.scan_history(), desc="Loading history"):
            single_run_complete_history.append(x)
        run_results[run.id] = {
            "id": run.id,
            "name": run.name,
            "history": single_run_complete_history,
        }
    return run_results


def convert_run_histories_to_df(
    metrics: List[str],
    aggregations: List[Union[str, Callable]],
    split_prefixes: str,
    resolution: Resolution,
    runs: Dict[str, Union[str, Dict[str, float]]],
    save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Configure relevant metrics which are relevant
    keys_of_interest = [
        f"{prefix}{k}{resolution.value}"
        for k in metrics
        for prefix in split_prefixes
    ]
    # Initialize dictionary to create pd.DataFrame from
    data = {k: [] for k in ["id"] + ["name"] + ["epoch"] + keys_of_interest}
    # Iterate over run_histories to format metrics
    for _, run in runs.items():
        run_history = run["history"]
        n = 0
        for k in keys_of_interest:
            append_values = [
                metrics_dict[k] for metrics_dict in run_history if metrics_dict[k]
            ]
            n = len(append_values)
            data[k] += append_values
        data["epoch"] += [i for i in range(n)]
        data["name"] += [run["name"] for _ in range(n)]
        data["id"] += [run["id"] for _ in range(n)]

    visualization_df = pd.DataFrame(data)
    if save_path:
        save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
    return (
        visualization_df,
        visualization_df.groupby("id").agg({k: aggregations for k in keys_of_interest}),
    )
