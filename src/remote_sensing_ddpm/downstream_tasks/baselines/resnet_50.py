from torch import nn
from torchvision.models import resnet50


def sigmoid_forward(self, x):
    x = self._forward_impl(x)
    return nn.Sigmoid(x)


def resnet50_variable_in_channels(in_channels: int, add_sigmoid: bool = False, *args, **kwargs):
    model = resnet50(*args, **kwargs)
    model.conv1 = nn.Conv2d(in_channels, model.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    if add_sigmoid:
        model.forward = sigmoid_forward
    return model
