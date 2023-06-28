import yaml
import copy
import wandb
from pathlib import Path
from typing import Any, Dict, Optional, List

# Lit Diffusion training loop and constants
from lit_diffusion.train import main
from lit_diffusion.constants import (
    PYTHON_KWARGS_CONFIG_KEY,
    TRAIN_TORCH_DATA_LOADER_CONFIG_KEY,
    VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY,
    PL_MODULE_CONFIG_KEY,
    PL_WANDB_LOGGER_CONFIG_KEY,
    SEED_CONFIG_KEY,
)

FE_CONFIG_KEY = "feature_extractor"
ADD_FE_KWARGS_CONFIG_KEY = "downstream_task_specific_feature_extractor_kwargs"
LABEL_PIPELINE_CONFIG_KEY = "label_pipeline"
PIPELINES_CONFIG_KEY = "pipelines"

LABEL_FRACTION_PATHS = {
    0.01: "/ds2/remote_sensing/ben-ge/ffcv/ben-ge-60-delta-multilabel-train-1-percent.beton",
    0.1: "/ds2/remote_sensing/ben-ge/ffcv/ben-ge-60-delta-multilabel-train-10-percent.beton",
    0.5: "/ds2/remote_sensing/ben-ge/ffcv/ben-ge-60-delta-multilabel-train-50-percent.beton",
}


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


def train(
    backbone_config: Dict,
    downstream_head_config: Dict,
    run_name: Optional[str] = None,
    repetition: Optional[int] = None,
):
    backbone_config = copy.deepcopy(backbone_config)
    downstream_head_config = copy.deepcopy(downstream_head_config)
    # Alter seed if part of multiple repetitions
    if repetition:
        downstream_head_config[SEED_CONFIG_KEY] += repetition

    # Join downstream task specific and regular key word arguments
    additional_kwargs = downstream_head_config.pop(ADD_FE_KWARGS_CONFIG_KEY)
    original_kwargs = backbone_config[FE_CONFIG_KEY][PYTHON_KWARGS_CONFIG_KEY]
    new_fe_kwargs = safe_join_dicts(additional_kwargs, original_kwargs)
    backbone_config[FE_CONFIG_KEY][PYTHON_KWARGS_CONFIG_KEY] = new_fe_kwargs

    # Add label-pipeline from downstream config
    label_pipeline_config: Dict = downstream_head_config[LABEL_PIPELINE_CONFIG_KEY]
    label_field = list(label_pipeline_config.keys())[0]
    for k, v in backbone_config[PIPELINES_CONFIG_KEY].items():
        # Check that the same field is not requested as both label and model input
        if v and k == label_field:
            print(f"Run ({run_name}) is being skipped since label field {label_field} is present in pipeline")
            return
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
    complete_config = safe_join_dicts(backbone_config, downstream_head_config)

    # Set wandb run name if it exists
    if run_name:
        complete_config[PL_WANDB_LOGGER_CONFIG_KEY]["name"] = run_name

    # run standard pytorch lightning training loop
    main(config=complete_config)
    wandb.finish()


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

    # Run the train function
    for b_name, b_config in backbone_configs.items():
        for dh_name, dh_config in downstream_head_configs.items():
            # Create run name based on config files name
            wandb_run_name = create_wandb_run_name(
                backbone_name=b_name,
                downstream_head_name=dh_name,
            )

            # Run Label Fraction experiments if desired
            if args.label_fractions.lower() == "true":
                for fraction, fraction_dataset_path in LABEL_FRACTION_PATHS.items():
                    lf_run_name = wandb_run_name + f"-lf-{fraction}"
                    print(f"Starting run {lf_run_name}")
                    lf_b_config = copy.deepcopy(b_config)
                    lf_b_config[TRAIN_TORCH_DATA_LOADER_CONFIG_KEY][PYTHON_KWARGS_CONFIG_KEY]["fname"] = fraction_dataset_path
                    train(
                        backbone_config=lf_b_config,
                        downstream_head_config=dh_config,
                        run_name=lf_run_name,
                    )

            print(f"Starting run {wandb_run_name}")
            for i in range(repetitions):
                train(
                    backbone_config=b_config,
                    downstream_head_config=dh_config,
                    run_name=wandb_run_name,
                    repetition=i,
                )
