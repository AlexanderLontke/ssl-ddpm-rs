from enum import Enum

# Configuration keys for instantiation and training loop
# Data
TORCH_DATA_LOADER_CONFIG_KEY = "torch_data_loader"
# Models
P_THETA_MODEL_CONFIG_KEY = "p_theta_model"
DDPM_CONFIG_KEY = "ddpm"
# Pytorch lightning
PL_TRAINER_CONFIG_KEY = "pl_trainer"
PL_WANDB_LOGGER_CONFIG_KEY = "pl_wandb_logger"
PL_MODEL_CHECKPOINT_CONFIG_KEY = "pl_checkpoint_callback"

# Fixed config keys for python module instantiation
STRING_PARAMS_CONFIG_KEY = "params"
PYTHON_CLASS_CONFIG_KEY = "module"


# Enum for DDPM target options
class DiffusionTarget(Enum):
    X_0 = "x_0"
    EPS = "eps"


# Metric keys
_TRAIN_PREFIX = "train/"
TRAINING_LOSS_METRIC_KEY = _TRAIN_PREFIX + "mse_loss"
