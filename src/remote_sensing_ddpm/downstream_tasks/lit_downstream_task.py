import copy
from typing import Any, Callable, Dict, Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from remote_sensing_ddpm.downstream_tasks.downstream_task_model import (
    DownstreamTaskModel,
)
from torch import nn

import pytorch_lightning as pl

# Lit diffusion
from lit_diffusion.diffusion_base.constants import (
    LOGGING_TRAIN_PREFIX,
    LOGGING_VAL_PREFIX,
)

def _copy_model_parameters(model: nn.Module):
    return [p.clone().detach().cpu() for p in model.parameters()]

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
        # Register feature extractor weights to assert that they remain unchanged at the end of training
        self.original_fe_weights = _copy_model_parameters(model=self.downstream_model.feature_extractor)

    def _train_val_step(
        self, batch, metrics_dict: Optional[Dict[str, Callable]], logging_prefix: str
    ):
        # Get data sample
        y = batch[self.target_key]
        batch_size, *_ = y.shape

        y_hat = self.downstream_model(batch)
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

    def on_fit_end(self) -> None:
        n = len(self.original_fe_weights)
        current_weights = _copy_model_parameters(model=self.downstream_model.feature_extractor)
        assert n == len(current_weights)
        assert (
            all(torch.equal(self.original_fe_weights[i], current_weights[i]) for i in range(n))
        ), "Backbone weights have changed during training"

    # def on_validation_start(self) -> None:
    #     self.pre_validation_weights = _copy_model_parameters(model=self.downstream_model.downstream_layer)
    #     return super().on_validation_epoch_start()
    
    # def on_validation_end(self) -> None:
    #     n = len(self.pre_validation_weights)
    #     current_weights = _copy_model_parameters(model=self.downstream_model.downstream_layer)
    #     assert n == len(current_weights)
    #     for i in range(n):
    #         print(torch.equal(self.pre_validation_weights[i], current_weights[i]))
    #     assert (
    #         all(torch.equal(self.pre_validation_weights[i], current_weights[i]) for i in range(n))
    #     ), "Backbone weights have changed during validation epoch"
