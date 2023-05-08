from typing import Dict

# Lit Diffusion training loop and constants
from lit_diffusion.train import main
from lit_diffusion.constants import PYTHON_KWARGS_CONFIG_KEY

FE_CONFIG_KEY = "feature_extractor"
ADD_FE_KWARGS_CONFIG_KEY = "downstream_task_specific_feature_extractor_kwargs"


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

    # fuse both dictionaries
    complete_config = safe_join_dicts(feature_extractor_config, downstream_task_config)

    # run standard pytorch lightning training loop
    main(config=complete_config)
