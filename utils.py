import torch
from torch import nn


"""
Module for useful torch utils, for training a model
"""


def get_sgd_optimizer(model: nn.Module, lr: float = 0.001, momentum: float = 0.9, weight_decay: float = 0):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum)
    return optimizer


def get_adam_optimizer(model: nn.Module, lr: float = 0.001, momentum: float = 0.9):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)
    return optimizer


def init_func__zero_mean_gaussian(std: float = 1):
    def func(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, a=0, b=std)
            m.bias.data.fill_(0.01)
    return func


def init_func__xavier():
    def func(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    return func
