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

def experiment_q3():
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


if __name__ == '__main__':
    experiment_q2()
