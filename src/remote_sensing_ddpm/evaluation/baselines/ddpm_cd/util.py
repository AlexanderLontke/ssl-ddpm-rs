import wandb


def set_option_from_sweep(wandb_config_key, option_value):
    if hasattr(wandb.config, wandb_config_key):
        return getattr(wandb.config, wandb_config_key)
    else:
        return option_value
