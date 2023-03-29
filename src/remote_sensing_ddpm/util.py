from typing import Dict

from importlib import import_module

from remote_sensing_ddpm.constants import (
    PYTHON_CLASS_CONFIG_KEY,
    STRING_PARAMS_CONFIG_KEY,
)


def instantiate_python_class_from_string_config(class_config: Dict):
    # Assert that necessary keys are contained in config
    assert all(
        k in class_config.keys()
        for k in [PYTHON_CLASS_CONFIG_KEY, STRING_PARAMS_CONFIG_KEY]
    )
    # Get module and class names
    module_full_name: str = class_config[PYTHON_CLASS_CONFIG_KEY]
    module_sub_names = module_full_name.split(".")
    module_name = ".".join(module_sub_names[:-1])
    class_name = module_sub_names[-1]
    # Import necessary module
    module = import_module(module_name)
    # Instantiate class with config values
    return getattr(module, class_name)(**class_config[STRING_PARAMS_CONFIG_KEY])