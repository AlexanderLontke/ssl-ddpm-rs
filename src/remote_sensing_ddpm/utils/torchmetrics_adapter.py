from typing import Any

import torch

from torchmetrics.metric import Metric


class TorchmetricsAdapter(Metric):
    def __init__(
        self,
        torchmetrics_module: Metric,
        apply_argmax: bool,
        device: str,
    ):
        super().__init__()
        self.torchmetrics_module = torchmetrics_module.to(torch.device(device))
        self.apply_argmax = apply_argmax
        self.__class__.__name__ = self.torchmetrics_module.__class__.__name__

    def _format_input(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.to(self.torchmetrics_module.device)
        targets = targets.to(self.torchmetrics_module.device)

        if self.apply_argmax:
            inputs = torch.argmax(inputs, dim=1)
        return inputs, targets

    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        inputs, targets = self._format_input(
            inputs=inputs, targets=targets
        )
        return self.torchmetrics_module.update(inputs, targets)

    def compute(self) -> Any:
        return self.torchmetrics_module.compute()


if __name__ == '__main__':
    import torch
    from torchmetrics.classification import MulticlassAccuracy
    from torchmetrics import ClasswiseWrapper
    metric = ClasswiseWrapper(TorchmetricsAdapter(
        torchmetrics_module=MulticlassAccuracy(num_classes=3, average=None),
        apply_argmax=False,
        device="cpu",
    ))
    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))
    print(metric(preds, target))
