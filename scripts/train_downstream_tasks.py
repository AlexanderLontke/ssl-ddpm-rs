from typing import Dict

# Lit Diffusion training loop and constants
from lit_diffusion.train import main
from lit_diffusion.constants import (
    PYTHON_KWARGS_CONFIG_KEY,
    TRAIN_TORCH_DATA_LOADER_CONFIG_KEY,
    VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY,
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


def train(feature_extractor_config: Dict, downstream_task_config: Dict):
    # Join downstream task specific and regular key word arguments
    additional_kwargs = downstream_task_config.pop(ADD_FE_KWARGS_CONFIG_KEY)
    original_kwargs = feature_extractor_config[FE_CONFIG_KEY][PYTHON_KWARGS_CONFIG_KEY]
    new_fe_kwargs = safe_join_dicts(additional_kwargs, original_kwargs)
    feature_extractor_config[FE_CONFIG_KEY][PYTHON_KWARGS_CONFIG_KEY] = new_fe_kwargs

    # Add label pipeline from downstream config
    label_pipeline_config: Dict = downstream_task_config[LABEL_PIPELINE_CONFIG_KEY]
    feature_extractor_config[PIPELINES_CONFIG_KEY].update(label_pipeline_config)

    # Get index to key mapping based on all FFCV pipelines that are not none
    mapping = []
    for k, v in feature_extractor_config[PIPELINES_CONFIG_KEY].items():
        if v:
            mapping.append(k)
    # Sort mapping (alphabetically) so that it matches with the return order of the FFCV dataset
    mapping = sorted(mapping)

    # Update new key word argument for both dataloaders
    for dataloader_key in (TRAIN_TORCH_DATA_LOADER_CONFIG_KEY, VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY):
        # 1. Update Pipelines
        feature_extractor_config[dataloader_key][
            PYTHON_KWARGS_CONFIG_KEY
        ].update(PIPELINES_CONFIG_KEY, feature_extractor_config[PIPELINES_CONFIG_KEY])
        # 2. Add mapping
        feature_extractor_config[dataloader_key][
            PYTHON_KWARGS_CONFIG_KEY
        ].update("mapping", mapping)

    # fuse both dictionaries
    complete_config = safe_join_dicts(feature_extractor_config, downstream_task_config)

    # run standard pytorch lightning training loop
    main(config=complete_config)
