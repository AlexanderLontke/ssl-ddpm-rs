from typing import Dict

# PyTorch imports
from torch.utils.data import DataLoader

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Util
from remote_sensing_ddpm.util import instantiate_python_class_from_string_config
from remote_sensing_ddpm.constants import (
    TORCH_DATA_LOADER_CONFIG_KEY,
    P_THETA_MODEL_CONFIG_KEY,
    DDPM_CONFIG_KEY,
    PL_TRAINER_CONFIG_KEY,
    PL_WANDB_LOGGER_CONFIG_KEY,
    PL_MODEL_CHECKPOINT_CONFIG_KEY
)
# Diffusion
from remote_sensing_ddpm.diffusion_process.ddpm import LitDDPM


def main(config: Dict):
    # Instantiate dataset
    train_dataset = ...
    # Instantiate dataloader from data set
    train_data_loader = DataLoader(
        dataset=train_dataset,
        **config[TORCH_DATA_LOADER_CONFIG_KEY]
    )
    # Instantiate model approximating p_theta(x_{t-1}|x_t)
    p_theta_model = instantiate_python_class_from_string_config(
        class_config=config[P_THETA_MODEL_CONFIG_KEY]
    )
    # TODO Instantiate DDPM class
    ddpm_pl_module = LitDDPM(
        p_theta_model=p_theta_model,
        **config[DDPM_CONFIG_KEY]
    )

    # PL-Trainer with the following features:
    # - Model Summary
    # - Progressbar
    # - WAND logging
    # - Checkpointing
    trainer = pl.Trainer(
        **config[PL_TRAINER_CONFIG_KEY],
        logger=WandbLogger(
            **config[PL_WANDB_LOGGER_CONFIG_KEY]
        ),
        callbacks=[
            ModelSummary(),
            ModelCheckpoint(
                **config[PL_MODEL_CHECKPOINT_CONFIG_KEY],
            ),
        ]
    )
    # Run training
    trainer.fit(
        model=ddpm_pl_module,
        train_dataloaders=train_data_loader,
    )
