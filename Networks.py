# LLM-assisted (Claude, Anthropic): network classes, make_activation, n_params and the examples.
from typing import cast

import torch
from torch import nn

from config import N_CLASSES









ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def make_activation(name: str, **kwargs): # feed as many at a time as you want, i hope :|
    return ACTIVATIONS[name](**kwargs)

import numpy as np
class LogisticRegressionNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        # z = x W^T + b with W (n_classes, n_features), b (n_classes,)
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(self.flatten(x).shape)
        return self.linear(self.flatten(x))   # (B, n_classes) logits


class FFNN(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...],
                 activation: str, n_classes: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(make_activation(activation))
        layers.append(nn.Linear(in_dim, n_classes))     # output logits no activation
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))
