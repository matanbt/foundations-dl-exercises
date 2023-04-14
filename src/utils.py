import torch
from torch import nn


"""
Module for useful torch utils, for training a model
"""


def get_sgd_optimizer(model: nn.Module, lr: float = 0.01, momentum: float = 0.4, weight_decay: float = 0):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum)
    print("model parameters: ", model.parameters())
    return optimizer


def get_adam_optimizer(model: nn.Module, lr: float = 0.01):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)
    return optimizer


def init_func__zero_mean_gaussian(std: float = 0.1):
    def func(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.uniform_(m.weight, a=0, b=std)
            m.bias.data.fill_(0.01)
    return func


def init_func__xavier():
    def func(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    return func
