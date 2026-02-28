import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation(name):
    """
    Повертає модуль функції активації на основі назви.
    Підтримує: ReLU, LeakyReLU, ELU, SELU, GELU, Swish, Mish, Hardswish, Softplus.
    """
    activations = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(inplace=True),
        "elu": nn.ELU(inplace=True),
        "selu": nn.SELU(inplace=True),
        "gelu": nn.GELU(),
        "swish": Swish(),
        "mish": Mish(),
        "hardswish": nn.Hardswish(inplace=True),
        "softplus": nn.Softplus(),
    }

    name = name.lower()
    if name in activations:
        return activations[name]
    else:
        raise ValueError(
            f"Activation function '{name}' not supported. Available: {list(activations.keys())}"
        )
