import pandas as pd
from torch import nn

from dataset import get_california_dataloaders
from nn_model_lit import get_lnn_regression_model


def experiment_q2():
    for N in [2, 3, 4]:
        trainloader, testloader, in_dim, out_dim = get_california_dataloaders(batch_size=512,
                                                                              full_batch_train=True)
        model, trainer = get_lnn_regression_model(in_dim, out_dim, N=N)

        model.train_dl = trainloader  # will be used for testing
        trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)

    """
    Options for getting results:
    (1) `df = pd.read_csv('./logs/lightning_logs/version_0/metrics.csv')`
    (2) inspecting Tensorboard artifacts `$ tensorboard --logdir tb_logs`
    """


def experiment_q3_1():
    for N in [2, 3]:
        trainloader, testloader, in_dim, out_dim = get_california_dataloaders(batch_size=512,
                                                                              full_batch_train=True)
        model, trainer = get_lnn_regression_model(in_dim, out_dim, N=N)

        model.train_dl = trainloader  # will be used for testing
        trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)

    """
    Options for getting results:
    (1) `df = pd.read_csv('./logs/lightning_logs/version_0/metrics.csv')`
    (2) inspecting Tensorboard artifacts `$ tensorboard --logdir tb_logs`
    """


def experiment_q3_2():
    width_to_diff = {}
    for width in [10**2, 10**3, 10**4, 10**5]:
        trainloader, testloader, in_dim, out_dim = get_california_dataloaders(batch_size=512,
                                                             full_batch_train=True)
        model, trainer = get_lnn_regression_model(in_dim, out_dim,
                                                  train_size=len(trainloader.dataset),
                                                  width=width,
                                                  activation_func=nn.ReLU(),
                                                  N=1)
        width_to_diff[width] = pd.read_csv('./logs/lightning_logs/version_0/metrics.csv')['logits_to_u_ntk_diff'].dropna()

        model.train_dl = trainloader  # will be used for testing
        trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)

    print(width_to_diff)
    pd.DataFrame(width_to_diff).to_csv('./results_exp_q3_2.csv')


if __name__ == '__main__':
    experiment_q3_2()
