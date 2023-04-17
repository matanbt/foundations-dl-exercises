from typing import Union

import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from dataclasses import dataclass, asdict
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.cifar10_dataset import trainloader, testloader
from src.models import BaselineNN
from src.utils import init_func__zero_mean_gaussian, get_sgd_optimizer


"""
Module for training a torch model
"""


# Scheme for training results object
@dataclass
class TrainResults:
    train_accuracies: np.array
    train_losses: np.array
    test_accuracies: np.array
    test_losses: np.array

    def get_accuracies_curve(self) -> go.Figure:
        fig = px.line(pd.DataFrame(asdict(self)) * 100,
                      y=['train_accuracies', 'test_accuracies'],
                      title="Accuracy over Epochs", labels={'index': 'Epoch', 'value': 'Accuracy (%)'})
        fig.update_layout(yaxis_range=[0, 105])
        return fig

    def get_losses_curve(self) -> go.Figure:
        fig = px.line(pd.DataFrame(asdict(self)),
                      y=['train_losses', 'test_losses'],
                      title="Losses over Epochs", labels={'index': 'Epoch', 'value': 'Loss'})
        fig.update_layout(yaxis_range=[0, 3])
        return fig

    def report(self):
        print(">> Accuracies Curves:")
        self.get_accuracies_curve().show()
        print(">> Losses Curves:")
        self.get_losses_curve().show()
        print(">> Optimization ended with:")
        print(f"   >> TRAIN-SET: best-accuracy={self.train_accuracies.max() * 100}, accuracy={self.train_accuracies[-1] * 100}%, loss={self.train_losses[-1] }")
        print(f"   >> TEST-SET: best-accuracy={self.test_accuracies.max() * 100} accuracy={self.test_accuracies[-1] * 100}%, loss={self.test_losses[-1] }")


class MultipleTrainResults:
    result_keys = None
    result_df   = None

    def __init__(self, result_dict): 
        self.result_keys = result_dict.keys()
        result_df_list = [pd.DataFrame(asdict(v)).add_prefix(f"{k}_") for (k,v) in result_dict.items()]
        self.result_df = pd.concat(result_df_list, axis=1)

    def get_accuracies_curve(self) -> go.Figure:
        fig = px.line(self.result_df * 100,
                      x=self.result_df.index,
                      y=[f"{k}_train_accuracies" for k in self.result_keys] + [f"{k}_test_accuracies" for k in self.result_keys],
                      title="Accuracy over Epochs", labels={'index': 'Epoch', 'value': 'Accuracy (%)'})
        fig.update_layout(yaxis_range=[0, 100])
        return fig

    def get_losses_curve(self) -> go.Figure:
        fig = px.line(self.result_df,
                      x=self.result_df.index,
                      y=[f"{k}_train_losses" for k in self.result_keys] + [f"{k}_test_losses" for k in self.result_keys],
                      title="Loss over Epochs", labels={'index': 'Epoch', 'value': 'Loss'})
        fig.update_layout(yaxis_range=[0, 3])
        return fig


# run training on a torch model
def train(model: nn.Module,
          trainloader: torch.utils.data.DataLoader,
          testloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          init_func: callable,  # function for weight initialization
          device: Union[torch.device, str] = 'cpu',
          num_epochs: int = 1):

    print(f">> Runs training of {model} on device={device} for {num_epochs} epochs.")

    # set device:
    model.to(device)

    # metrics:
    train_losses = np.zeros(num_epochs)
    train_accuracies = np.zeros(num_epochs)
    test_losses = np.zeros(num_epochs)
    test_accuracies = np.zeros(num_epochs)

    # init weights
    model.apply(init_func)

    # Training loop
    for epoch in range(num_epochs):

        # Training Epoch
        model.train()
        with tqdm(trainloader, unit="batch", desc=f"Train - Epoch {epoch}") as tepoch:
            for x_batch, y_batch in tepoch:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Clear previous gradients calc
                optimizer.zero_grad()

                # Forward
                logits = model(x_batch)

                # Backward: Calculate loss and back-prop
                loss = model.loss_fn(logits, y_batch)
                
                # print("x: ", x_batch.shape)
                # print("y: ", y_batch.shape)
                # print("logits: ", logits.shape)
                # print("loss: ", loss.shape)

                loss.backward()
                optimizer.step()

                # Record train metrics
                train_losses[epoch] += loss.item()
                train_accuracies[epoch] += (logits.argmax(dim=-1) == y_batch).to(float).mean().item()

                tepoch.set_postfix(loss=loss.item(),
                                   accuracy=(logits.argmax(dim=-1) == y_batch).to(float).mean().item())
            # normalize summed losses and accuracies (avg over batches)
            train_accuracies[epoch] /= len(trainloader)
            train_losses[epoch] /= len(trainloader)
            tepoch.set_postfix(loss=train_accuracies[epoch], accuracy=train_losses[epoch])

        # Evaluation Epoch
        model.eval()
        with torch.no_grad():
            with tqdm(testloader, unit="batch", desc=f"Evaluate - Epoch {epoch}") as tepoch:
                for x_batch, y_batch in tepoch:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                    # Forward pass
                    logits = model(x_batch)

                    # Calculate the loss
                    loss = model.loss_fn(logits, y_batch)

                    # Record test metrics
                    test_losses[epoch] += loss.item()
                    test_accuracies[epoch] += (logits.argmax(dim=-1) == y_batch).to(float).mean().item()

                    tepoch.set_postfix(loss=loss.item(),
                                       accuracy=(logits.argmax(dim=-1) == y_batch).to(float).mean().item())
            # normalize
            test_accuracies[epoch] /= len(testloader)
            test_losses[epoch] /= len(testloader)
            tepoch.set_postfix(loss=test_losses[epoch], accuracy=test_accuracies[epoch])

    return TrainResults(
        train_accuracies=train_accuracies,
        train_losses=train_losses,
        test_accuracies=test_accuracies,
        test_losses=test_losses,
    )


if __name__ == '__main__':
    model = BaselineNN()
    results: TrainResults = train(
        model=model,
        init_func=init_func__zero_mean_gaussian(std=0.1),
        optimizer=get_sgd_optimizer(model, lr=0.01, momentum=0.4),
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=3,
        device='cpu'
    )
    results.get_accuracies_curve().show()

