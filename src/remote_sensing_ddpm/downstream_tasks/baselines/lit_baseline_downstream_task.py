import copy
from typing import Any, Callable, Dict, Optional, List

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from remote_sensing_ddpm.downstream_tasks.downstream_task_model import (
    DownstreamTaskModel,
)
from torch import nn
from torchmetrics.metric import Metric

import pytorch_lightning as pl

# Lit diffusion
from lit_diffusion.diffusion_base.constants import (
    LOGGING_TRAIN_PREFIX,
    LOGGING_VAL_PREFIX,
)


class LitBaselineDownstreamTask(pl.LightningModule):
    def __init__(
            self,
            downstream_model: DownstreamTaskModel,
            learning_rate: float,
            loss: nn.Module,
            target_key: str,
            data_key: str,
            stack_input_keys: List[str] = None,
            training_metrics: Optional[Dict[str, Metric]] = None,
            validation_metrics: Optional[Dict[str, Metric]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.downstream_model = downstream_model
        self.target_key = target_key
        self.data_key = data_key

        # Stack inputs for keys
        self.stack_input_keys = stack_input_keys

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

        x = batch[self.data_key]
        if self.stack_input_keys is not None:
            for k in self.stack_input_keys:
                x = torch.cat([x, batch[k]], dim=1)

        y_hat = self.downstream_model(x)
        step_loss = self.loss(y_hat, y)

        # Log trainings loss
        self.log(
            logging_prefix + type(self.loss).__name__,
            step_loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Log any additional metrics
        if metrics_dict:
            for metric_name, metric_function in metrics_dict.items():
                if hasattr(metric_function, "device") and hasattr(
                        metric_function, "to"
                ):
                    metric_function = metric_function.to(y_hat.device)
                logging_kwargs = {
                    # "on_step": True,
                    # "on_epoch": True,
                    "batch_size": batch_size,
                }
                value = metric_function(y_hat, y)
                if isinstance(value, Dict):
                    self.log_dict(
                        dictionary={logging_prefix + k: v for k, v in value.items()},
                        **logging_kwargs,
                    )
                else:
                    self.log(
                        name=logging_prefix + metric_name, value=value, **logging_kwargs
                    )
        return step_loss

    def training_step(
            self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        return self._train_val_step(
            batch=batch,
            metrics_dict=self.training_metrics,
            logging_prefix=LOGGING_TRAIN_PREFIX,
        )

    def on_train_epoch_end(self) -> None:
        for _, metric in self.training_metrics.items():
            metric.reset()

    def validation_step(
            self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        return self._train_val_step(
            batch=batch,
            metrics_dict=self.validation_metrics,
            logging_prefix=LOGGING_VAL_PREFIX,
        )

    def on_validation_epoch_end(self) -> None:
        for _, metric in self.validation_metrics.items():
            metric.reset()

    def test_step(
            self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        return self._train_val_step(
            batch=batch,
            metrics_dict=self.validation_metrics,
            logging_prefix="test/",
        )

    def configure_optimizers(self) -> Any:
        return_dict = {}
        lr = self.learning_rate
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        return_dict["optimizer"] = opt
        return return_dict
