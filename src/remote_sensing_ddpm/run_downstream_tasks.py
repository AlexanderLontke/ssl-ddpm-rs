import os

import yaml
import copy
import wandb
from pathlib import Path
from typing import Any, Dict, Optional, List

# Pandas
import pandas as pd

# PyTorch
import torch

# Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

# Lit Diffusion training loop and constants
from lit_diffusion.train import main
from lit_diffusion.util import instantiate_python_class_from_string_config
from lit_diffusion.constants import (
    PYTHON_KWARGS_CONFIG_KEY,
    TRAIN_TORCH_DATA_LOADER_CONFIG_KEY,
    VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY,
    PL_MODULE_CONFIG_KEY,
    PL_WANDB_LOGGER_CONFIG_KEY,
    SEED_CONFIG_KEY,
    DEVICE_CONFIG_KEY,
    PL_TRAINER_CONFIG_KEY,
)

# ARGPARSE OPTIONS
MODE_OPTION_TRAIN = "train"
MODE_OPTION_TEST = "test"

# LOGGING KEYS
EPOCH_KEY = "epoch"
LAST_CKPT_NAME = "last.ckpt"

# CONFIG KEYS
FE_CONFIG_KEY = "feature_extractor"
ADD_FE_KWARGS_CONFIG_KEY = "downstream_task_specific_feature_extractor_kwargs"
LABEL_PIPELINE_CONFIG_KEY = "label_pipeline"
PIPELINES_CONFIG_KEY = "pipelines"
MONITOR_KEY = "monitor"
MONITOR_MODE_KEY = "monitor_mode"
PROJECT_KEY = "name"

# Project specific values which should be dynamic instead of constants but corners were cut here
LABEL_FRACTION_PATHS = {
    0.01: "/ds2/remote_sensing/ben-ge/ffcv/ben-ge-60-delta-multilabel-train-1-percent.beton",
    0.1: "/ds2/remote_sensing/ben-ge/ffcv/ben-ge-60-delta-multilabel-train-10-percent.beton",
    0.5: "/ds2/remote_sensing/ben-ge/ffcv/ben-ge-60-delta-multilabel-train-50-percent.beton",
}
AVAILABLE_CHECKPOINT_EPOCHS = [4, 9, 14, 19]
WANDB_PROJECT_NAME = "ssl-diffusion"


# UTIL METHODS
def safe_join_dicts(dict_a: Dict, dict_b: Dict) -> Dict:
    for x in dict_b.keys():
        if x in dict_a.keys():
            assert (
                dict_a[x] == dict_b[x]
            ), f"Difference at key {x} with values {dict_a[x]} != {dict_b[x]}"
    for x in dict_a.keys():
        if x in dict_b.keys():
            assert (
                dict_a[x] == dict_b[x]
            ), f"Difference at key {x} with values {dict_a[x]} != {dict_b[x]}"
    return {**dict_a, **dict_b}


def fuse_backbone_and_downstream_head_config(
    backbone_config: Dict,
    downstream_head_config: Dict,
):
    # Join downstream task specific and regular key word arguments
    additional_kwargs = downstream_head_config.pop(ADD_FE_KWARGS_CONFIG_KEY)
    original_kwargs = backbone_config[FE_CONFIG_KEY][PYTHON_KWARGS_CONFIG_KEY]
    new_fe_kwargs = safe_join_dicts(additional_kwargs, original_kwargs)
    backbone_config[FE_CONFIG_KEY][PYTHON_KWARGS_CONFIG_KEY] = new_fe_kwargs

    # Add label-pipeline from downstream config
    if LABEL_PIPELINE_CONFIG_KEY in downstream_head_config.keys():
        label_pipeline_config: Dict = downstream_head_config[LABEL_PIPELINE_CONFIG_KEY]
        label_field = list(label_pipeline_config.keys())[0]
        for k, v in backbone_config[PIPELINES_CONFIG_KEY].items():
            # Check that the same field is not requested as both label and model input
            if v and k == label_field:
                return None
        backbone_config[PIPELINES_CONFIG_KEY].update(label_pipeline_config)

        # Get index to key mapping based on all FFCV pipelines that are not none
        mapping = []
        for k, v in backbone_config[PIPELINES_CONFIG_KEY].items():
            if v:
                mapping.append(k)

        # Sort mapping (alphabetically) so that it matches with the return order of the FFCV dataset
        mapping = sorted(mapping)

        # Update new key word argument for both dataloaders
        dataloader_keys = (
            [TRAIN_TORCH_DATA_LOADER_CONFIG_KEY]
            if VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY not in backbone_config.keys()
            else [
                TRAIN_TORCH_DATA_LOADER_CONFIG_KEY,
                VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY,
            ]
        )
        for dataloader_key in dataloader_keys:
            # 1. Update Pipelines
            backbone_config[dataloader_key][PYTHON_KWARGS_CONFIG_KEY].update(
                {PIPELINES_CONFIG_KEY: backbone_config[PIPELINES_CONFIG_KEY]}
            )
            # 2. Add mapping
            backbone_config[dataloader_key][PYTHON_KWARGS_CONFIG_KEY].update(
                {"mapping": mapping}
            )

    # Fuse both dictionaries
    feature_extractor_config = backbone_config.pop(FE_CONFIG_KEY)
    downstream_head_config[PL_MODULE_CONFIG_KEY][PYTHON_KWARGS_CONFIG_KEY][
        "downstream_model"
    ][PYTHON_KWARGS_CONFIG_KEY][FE_CONFIG_KEY] = feature_extractor_config
    return safe_join_dicts(backbone_config, downstream_head_config)


def read_yaml_config_file_or_dir(config_file_path: Path) -> Dict[str, Any]:
    read_configs = {}
    # handle path-is-file case
    if config_file_path.is_file():
        with config_file_path.open("r") as config_file:
            read_configs[config_file_path.name] = yaml.safe_load(config_file)
    # handle path-is-directory case
    elif config_file_path.is_dir():
        # Read in all yaml files in directory
        config_file_paths = config_file_path.glob("*.yaml")
        for config_file_path in config_file_paths:
            with config_file_path.open("r") as config_file:
                read_configs[config_file_path.name] = yaml.safe_load(config_file)
    else:
        raise ValueError(
            f"Backbone config path ({config_file_path}) was neither a file nor a directory"
        )
    return read_configs


def create_wandb_run_name(backbone_name: str, downstream_head_name: str) -> str:
    return "-".join([pn.split(".")[0] for pn in (backbone_name, downstream_head_name)])


# Loop Methods
def train(
    complete_config: Dict,
    run_name: Optional[str] = None,
    repetition: Optional[int] = None,
):
    complete_config = copy.deepcopy(complete_config)
    # Alter seed if part of multiple repetitions
    if repetition:
        complete_config[SEED_CONFIG_KEY] += repetition

    # Set wandb run name if it exists
    if run_name:
        complete_config[PL_WANDB_LOGGER_CONFIG_KEY]["name"] = run_name

    # run standard pytorch lightning training loop
    main(config=complete_config)
    wandb.finish()


def get_best_epoch_through_wandb(
    wandb_run,
    checkpoint_path: Path,
    keys_of_interest: List[str],
    monitor: str,
    monitor_mode: str,
) -> Path:
    single_run_complete_history = []
    for x in wandb_run.scan_history(keys=keys_of_interest, page_size=10000):
        single_run_complete_history.append(x)
    history_df = pd.DataFrame(single_run_complete_history)
    history_df = history_df.loc[AVAILABLE_CHECKPOINT_EPOCHS, :]
    best_epoch = getattr(history_df[monitor], f"idx{monitor_mode}")()
    if best_epoch == AVAILABLE_CHECKPOINT_EPOCHS[-1]:
        file_name = LAST_CKPT_NAME
    else:
        file_name = [
            file_name
            for file_name in os.listdir(checkpoint_path)
            if file_name.startswith(f"epoch={int(best_epoch)}")
        ][0]
    return checkpoint_path / file_name


def get_best_epoch_through_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_callback_states = torch.load(checkpoint_path / LAST_CKPT_NAME)[
        "callbacks"
    ]
    checkpoint_callback_keys = [
        k for k in checkpoint_callback_states.keys() if k.startswith("ModelCheckpoint")
    ]
    assert len(checkpoint_callback_keys) == 1
    checkpoint_callback_key = checkpoint_callback_keys[0]
    checkpoint_callback_state = checkpoint_callback_states[checkpoint_callback_key]
    return Path(checkpoint_callback_state["best_model_path"])


def get_best_checkpoints(
    wandb_run_name: str, wandb_sub_project_name: str, through_callback: bool = False
) -> List[Path]:
    # This is only necessary because the mode of the checkpoint callback was misconfigured
    api = wandb.Api()
    run_filter = {
        "$and": [
            {"display_name": {"$eq": wandb_run_name}},
            {"state": {"$eq": "finished"}},
        ]
    }
    monitor = complete_config[MONITOR_KEY]
    monitor_mode = complete_config[MONITOR_MODE_KEY]
    runs = [
        run
        for run in api.runs(
            f"{WANDB_PROJECT_NAME}/{wandb_sub_project_name}", filters=run_filter
        )
    ]
    keys_of_interest = [EPOCH_KEY, monitor]

    best_checkpoint_paths = []
    # Get all data
    for run in runs:
        checkpoint_path = Path(f"{wandb_sub_project_name}/{run.id}/checkpoints/")
        if through_callback:
            checkpoint_path = get_best_epoch_through_checkpoint(
                checkpoint_path=checkpoint_path
            )
        else:
            checkpoint_path = get_best_epoch_through_wandb(
                wandb_run=run,
                checkpoint_path=checkpoint_path,
                keys_of_interest=keys_of_interest,
                monitor=monitor,
                monitor_mode=monitor_mode,
            )
        best_checkpoint_paths.append(checkpoint_path)
    return best_checkpoint_paths


def run_test(complete_config: Dict, test_beton_file: Path, wandb_run_name: str, through_callback: bool):
    # Fetch best checkpoints
    wandb_sub_project_name = complete_config[PROJECT_KEY]
    best_checkpoints = get_best_checkpoints(wandb_run_name, wandb_sub_project_name, through_callback=through_callback)
    eval_suffix = "-eval"
    wandb_project_name = complete_config[PL_WANDB_LOGGER_CONFIG_KEY]["project"]
    complete_config[PL_WANDB_LOGGER_CONFIG_KEY]["project"] = (
        wandb_project_name + eval_suffix
        if not wandb_project_name.endswith(eval_suffix)
        else wandb_project_name
    )
    complete_config[PL_WANDB_LOGGER_CONFIG_KEY]["name"] = (
        wandb_run_name + eval_suffix
        if not wandb_run_name.endswith(eval_suffix)
        else wandb_run_name
    )
    original_config = copy.deepcopy(complete_config)
    for checkpoint_path in best_checkpoints:
        print("Running eval for", checkpoint_path)
        complete_config = copy.deepcopy(original_config)
        wandb_logger = WandbLogger(
            **complete_config[PL_WANDB_LOGGER_CONFIG_KEY], config=complete_config
        )
        pl_module = instantiate_python_class_from_string_config(
            class_config=complete_config[PL_MODULE_CONFIG_KEY]
        )
        pl_module = pl_module.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=complete_config[DEVICE_CONFIG_KEY],
            downstream_model=pl_module.downstream_model,
            learning_rate=pl_module.learning_rate,
            loss=pl_module.loss,
            target_key=pl_module.target_key,
            validation_metrics=pl_module.validation_metrics,
        )
        # Create test dataloader config
        test_dataloader_config = copy.deepcopy(
            complete_config[VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY]
        )
        test_dataloader_config[PYTHON_KWARGS_CONFIG_KEY]["fname"] = test_beton_file
        test_dataloader = instantiate_python_class_from_string_config(
            class_config=test_dataloader_config
        )
        trainer = pl.Trainer(
            logger=wandb_logger,
            **complete_config[PL_TRAINER_CONFIG_KEY],
        )
        trainer.test(model=pl_module, dataloaders=test_dataloader)
        wandb.finish()


if __name__ == "__main__":
    import argparse

    # Add run arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--backbone-config",
        type=Path,
        help="Path to feature extractor config yaml",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--downstream-head-config",
        type=Path,
        help="Path to downstream head config yaml",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--training-repetitions",
        type=int,
        help="Number of times training should be repeated",
        default=1,
        required=False,
    )
    parser.add_argument(
        "-l",
        "--label-fractions",
        type=str,
        help="Flag showing whether or not to run label fraction Experiments",
        default="",
        required=False,
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Mode determining whether to train or test downstream models",
        default=MODE_OPTION_TRAIN,
        required=False,
    )

    parser.add_argument(
        "-t",
        "--test-beton-file",
        type=str,
        help="Beton file containing test dataset in FFCV format",
        required=False,
    )

    parser.add_argument(
        "--through-callback",
        type=str,
        default="True",
        help="determines how the best checkpoint is selected",
        required=False,
    )

    # Parse run arguments
    args = parser.parse_args()

    # Load backbone config file
    b_config_file_path = args.backbone_config
    backbone_configs = read_yaml_config_file_or_dir(b_config_file_path)

    # Load downstream task config file
    dh_config_file_path = args.downstream_head_config
    downstream_head_configs = read_yaml_config_file_or_dir(dh_config_file_path)

    # Get number of repetitions from
    repetitions = args.training_repetitions

    # Get mode
    mode = args.mode

    # Get best checkpoint method
    through_callback = args.through_callback.lower() in ["true"]

    # Run the train function
    for b_name, b_config in backbone_configs.items():
        for dh_name, dh_config in downstream_head_configs.items():
            # Create run name based on config files name
            wandb_run_name = create_wandb_run_name(
                backbone_name=b_name,
                downstream_head_name=dh_name,
            )

            # fuse configs
            backbone_config = copy.deepcopy(b_config)
            downstream_head_config = copy.deepcopy(dh_config)
            complete_config = fuse_backbone_and_downstream_head_config(
                backbone_config=backbone_config,
                downstream_head_config=downstream_head_config,
            )

            if complete_config is None:
                print(
                    f"Run ({wandb_run_name}) is being skipped since label field is present in pipeline"
                )
                continue

            # Run Label Fraction experiments if desired
            if args.label_fractions.lower() == "true":
                for fraction, fraction_dataset_path in LABEL_FRACTION_PATHS.items():
                    lf_run_name = wandb_run_name + f"-lf-{fraction}"
                    lf_config = copy.deepcopy(complete_config)
                    lf_config[TRAIN_TORCH_DATA_LOADER_CONFIG_KEY][
                        PYTHON_KWARGS_CONFIG_KEY
                    ]["fname"] = fraction_dataset_path
                    # TODO clean this up
                    if mode.lower() == MODE_OPTION_TRAIN:
                        print(f"Starting run {lf_run_name}")
                        train(
                            complete_config=lf_config,
                            run_name=lf_run_name,
                        )
                    elif mode.lower() == MODE_OPTION_TEST:
                        test_beton_file = args.test_beton_file
                        assert (
                            test_beton_file
                        ), "In testing mode a test beton file needs to be given use -t"
                        run_test(
                            complete_config=complete_config,
                            test_beton_file=test_beton_file,
                            wandb_run_name=lf_run_name,
                            through_callback=through_callback
                        )
                    else:
                        raise NotImplementedError(
                            f"mode option: {mode} not implemented"
                        )

            if mode.lower() == MODE_OPTION_TRAIN:
                for i in range(repetitions):
                    print(f"Starting run {wandb_run_name}")
                    train(
                        complete_config=complete_config,
                        run_name=wandb_run_name,
                        repetition=i,
                    )
            elif mode.lower() == MODE_OPTION_TEST:
                test_beton_file = args.test_beton_file
                assert (
                    test_beton_file
                ), "In testing mode a test beton file needs to be given use -t"
                run_test(
                    complete_config=complete_config,
                    test_beton_file=test_beton_file,
                    wandb_run_name=wandb_run_name,
                    through_callback=through_callback,
                )
            else:
                raise NotImplementedError(f"mode option: {mode} not implemented")
