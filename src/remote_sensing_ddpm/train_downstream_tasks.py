from typing import Dict, Optional

# Lit Diffusion training loop and constants
from lit_diffusion.train import main
from lit_diffusion.constants import (
    PYTHON_KWARGS_CONFIG_KEY,
    TRAIN_TORCH_DATA_LOADER_CONFIG_KEY,
    VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY,
    PL_MODULE_CONFIG_KEY,
    PL_WANDB_LOGGER_CONFIG_KEY,
)

FE_CONFIG_KEY = "feature_extractor"
ADD_FE_KWARGS_CONFIG_KEY = "downstream_task_specific_feature_extractor_kwargs"
LABEL_PIPELINE_CONFIG_KEY = "label_pipeline"
PIPELINES_CONFIG_KEY = "pipelines"


def safe_join_dicts(dict_a: Dict, dict_b: Dict) -> Dict:
    for x in dict_b.keys():
        if x in dict_a.keys():
            assert dict_a[x] == dict_b[x]
    for x in dict_a.keys():
        if x in dict_b.keys():
            assert dict_a[x] == dict_b[x]
    return {**dict_a, **dict_b}


def train(backbone_config: Dict, downstream_head_config: Dict, run_name: Optional[str] = None):
    # Join downstream task specific and regular key word arguments
    additional_kwargs = downstream_head_config.pop(ADD_FE_KWARGS_CONFIG_KEY)
    original_kwargs = backbone_config[FE_CONFIG_KEY][PYTHON_KWARGS_CONFIG_KEY]
    new_fe_kwargs = safe_join_dicts(additional_kwargs, original_kwargs)
    backbone_config[FE_CONFIG_KEY][PYTHON_KWARGS_CONFIG_KEY] = new_fe_kwargs

    # Add label pipeline from downstream config
    label_pipeline_config: Dict = downstream_head_config[LABEL_PIPELINE_CONFIG_KEY]
    backbone_config[PIPELINES_CONFIG_KEY].update(label_pipeline_config)

    # Get index to key mapping based on all FFCV pipelines that are not none
    mapping = []
    for k, v in backbone_config[PIPELINES_CONFIG_KEY].items():
        if v:
            mapping.append(k)
    # Sort mapping (alphabetically) so that it matches with the return order of the FFCV dataset
    mapping = sorted(mapping)

    # Update new key word argument for both dataloaders
    for dataloader_key in (
        TRAIN_TORCH_DATA_LOADER_CONFIG_KEY,
        VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY,
    ):
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


if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path

    # Add run arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--backbone-config",
        type=Path,
        help="Path to feature extractor config yaml",
        required=False,
    )

    parser.add_argument(
        "-d",
        "--downstream-head-config",
        type=Path,
        help="Path to downstream head config yaml",
        required=False,
    )

    # Parse run arguments
    args = parser.parse_args()

    # Load backbone config file
    b_config_file_path = args.backbone_config
    with b_config_file_path.open("r") as b_config_file:
        b_config = yaml.safe_load(b_config_file)

    # Load downstream task config file
    dh_config_file_path = args.downstream_head_config
    with dh_config_file_path.open("r") as dh_config_file:
        dh_config = yaml.safe_load(dh_config_file)

    # Create run name based on config files name
    wandb_run_name = "-".join([p.name.split(".")[0] for p in (b_config_file_path, dh_config_file_path)])

    # Run main function
    train(
        backbone_config=b_config, downstream_head_config=dh_config, run_name=wandb_run_name,
    )
