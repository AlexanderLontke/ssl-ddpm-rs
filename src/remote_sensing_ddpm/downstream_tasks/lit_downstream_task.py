from typing import Any, Callable, Dict, Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from remote_sensing_ddpm.downstream_tasks.downstream_models.downstream_task_model import DownstreamTaskModel
from torch import nn

import pytorch_lightning as pl

# Lit diffusion
from lit_diffusion.constants import LOGGING_TRAIN_PREFIX, LOGGING_VAL_PREFIX


class LitDownstreamTask(pl.LightningModule):
    def __init__(
        self,
        downstream_model: DownstreamTaskModel,
        learning_rate: float,
        loss: nn.Module,
        target_key: str,
        training_metrics: Optional[Dict[str, Callable]] = None,
        validation_metrics: Optional[Dict[str, Callable]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.downstream_model = downstream_model
        self.target_key = target_key

        # Setup learning rate
        self.learning_rate = learning_rate
        # Setup loss
        self.loss = loss
        # Setup metrics
        self.training_metrics = training_metrics
        self.validation_metrics = validation_metrics

    def _train_val_step(
        self, batch, metrics_dict: Optional[Dict[str, Callable]], logging_prefix: str
    ):
        # Get data sample
        y = batch[self.target_key]
        batch_size, *_ = y.shape

        y_hat = self.downstream_model(batch)
        loss = self.loss(y_hat, y)

        # Log trainings loss
        self.log(
            logging_prefix + str(loss.__name__),
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Log any additional metrics
        if metrics_dict:
            for metric_name, metric_function in metrics_dict.items():
                self.log(
                    name=logging_prefix + metric_name,
                    value=metric_function(y_hat, y),
                    on_epoch=True,
                    batch_size=batch_size,
                )
        return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        return self._train_val_step(
            batch=batch,
            metrics_dict=self.training_metrics,
            logging_prefix=LOGGING_TRAIN_PREFIX,
        )

    def validation_step(
        self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        return self._train_val_step(
            batch=batch,
            metrics_dict=self.validation_metrics,
            logging_prefix=LOGGING_VAL_PREFIX,
        )

    def configure_optimizers(self) -> Any:
        return_dict = {}
        lr = self.learning_rate
        params = list(self.downstream_model.get_parameters_to_optimize())
        opt = torch.optim.Adam(params, lr=lr)
        return_dict["optimizer"] = opt
        return return_dict
