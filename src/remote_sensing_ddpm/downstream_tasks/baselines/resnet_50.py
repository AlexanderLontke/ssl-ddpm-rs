from torch import nn
from torchvision.models import resnet50




def resnet50_variable_in_channels(in_channels: int, add_sigmoid: bool = False, *args, **kwargs):
    model = resnet50(*args, **kwargs)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.new_sigmoid = nn.Sigmoid()
    if add_sigmoid:#
        def sigmoid_forward(x):
            x = model._forward_impl(x)
            return model.new_sigmoid(x)
        model.forward = sigmoid_forward
    return model
