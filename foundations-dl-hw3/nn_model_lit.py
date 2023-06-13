import itertools
from typing import Tuple, List

import numpy as np
from scipy.linalg import fractional_matrix_power as frac_pow

import pandas as pd
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.functional import f1_score


class LitNNModel(LightningModule):
    def __init__(self,
                 # input-output dims
                 n_features: int = 9,
                 n_classes: int = 2,

                 # arch. hyperparameters
                 hidden_layers_dim: int = 50,
                 hidden_layers_count: int = 3,
                 hidden_layers_list: List[int] = None,  # if not None overrides the previous
                 activation_func: torch.nn.Module = torch.nn.ReLU,
                 loss_func: torch.nn.Module = nn.MSELoss,

                 # opt. hyperparameters
                 lr: float = 0.01,
                 weight_decay: float = 0.005,

                 # misc
                 train_size: int = 512,
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.loss = loss_func
        self.activation = activation_func

        # build model:
        layers = []
        if hidden_layers_list is None:
            hidden_layers_list = [hidden_layers_dim] * hidden_layers_count
        prev_output_dim = n_features
        for curr_hidden_dim in hidden_layers_list:  # append hidden layers
            layers += [
                torch.nn.Linear(prev_output_dim, curr_hidden_dim),
                self.activation,
            ]
            prev_output_dim = curr_hidden_dim
        layers += [  # append final layer
            torch.nn.Linear(prev_output_dim, n_classes),
        ]
        self.model = torch.nn.Sequential(*layers)

        # ----- model & other data for the gradient-flow-based case [PART3.EXP1] ----- #
        self.gf_model = torch.nn.Linear(n_features, n_classes)
        self.gf_loss = loss_func
        self.eta = lr
        self.N = len([layer for layer in self.model if hasattr(layer, "weight")])
        with torch.no_grad():
            self.gf_model.weight.copy_(self.get_e2e_mat())

        # ----- model & other data for the 'Neural Tangent Kernel' case [PART3.EXP2] ----- #
        self.ntk_u = torch.zeros((train_size, 1))  # the u being optimized via the NTK dynamics showed in class
        

    def forward(self, x):
        out = self.model(x)
        return out

    def get_e2e_mat(self):
        e2e_mat = None

        for layer in self.model:
            if hasattr(layer, "weight"):
                if e2e_mat == None:
                    e2e_mat = layer.weight
                else:
                    e2e_mat = layer.weight @ e2e_mat

        return e2e_mat

    def _update_u_ntk(self, x, y):
        """ NTK update rule [PART3.EXP2]"""

        normalized_x = nn.functional.normalize(x, dim=1)

        # x of shape (train_size, dim_in), so (x@x.T)_{i,j} contains <x_i,x_j>
        x_products = x @ x.T
        x_products_normalized = normalized_x @ normalized_x.T

        # H = (1/torch.pi)(X@X.t())*(torch.pi-torch.acos(normalized_X@normalized_X.t()))

        h_star = x_products * (1 / (2 * torch.pi)) * (torch.pi - torch.arccos(x_products_normalized))
        h_star = torch.nan_to_num(h_star, nan=0.0)  # replace nans, caused by h_star calculated
        update_u = - h_star @ (self.ntk_u - y)

        self.ntk_u += self.hparams.lr * update_u
        return self.ntk_u

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss(logits, y)
        self.evaluate(batch, logits=logits, stage='train')

        # ----- compute the step of the gradient-flow linear model [PART3.EXP1] ----- #

        # get ∇ℓ(Wt)
        self.gf_model.zero_grad()
        gf_logits = self.gf_model(x)
        gf_loss = self.gf_loss(gf_logits, y)
        gf_loss.backward()
        gf_loss_grad = self.gf_model.weight.grad

        # get Wt
        Wt  = self.gf_model.weight.data
        wwt = Wt @ Wt.T
        wtw = Wt.T @ Wt

        # update gf_model's weights to W_(t+1)
        eta, N = self.eta, self.N
        for j in range(1,N+1):
            self.gf_model.weight.data -= eta * torch.Tensor(frac_pow(wwt, (j-1)/N)) @ \
                                         gf_loss_grad.float() @ \
                                         torch.Tensor(frac_pow(wtw, (N-j)/N))

        # log results
        norm_of_trajectory_diff = torch.linalg.norm(self.gf_model.weight.data - self.get_e2e_mat())
        self.log("norm_of_trajectory_diff", norm_of_trajectory_diff, prog_bar=False, on_epoch=True, on_step=False)

        # ----- compute the step of the gradient-flow linear model [PART3.EXP2] ----- #
        self.ntk_u = self._update_u_ntk(x, y)

        # compare regularly-optimized model's logits to those of NTK
        logits_to_u_ntk_diff = torch.linalg.norm(logits - self.ntk_u)
        self.log("logits_to_u_ntk_diff", logits_to_u_ntk_diff, prog_bar=False, on_epoch=True, on_step=False)

        return loss

    def on_before_optimizer_step(self, optimizer):
        """called after loss.backward() and before optimizers are stepped., to inspect weights/gradients/etc."""

        # Gradient magnitude:
        flattened_grad = torch.cat([p.grad.flatten() for p in self.parameters()])
        grad_magnitude = torch.norm(flattened_grad)
        self.log(f"grad_magnitude", grad_magnitude, prog_bar=False, on_epoch=True, on_step=False)

        # Hessian values:
        # function from https://github.com/noahgolmant/pytorch-hessian-eigenthings
        from hessian_eigenthings import compute_hessian_eigenthings
        eigenvals, _ = compute_hessian_eigenthings(self.model, self.train_dl, self.loss,
                                                   sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                                                   use_gpu=False)  # TODO verify this works as expected
        self.log(f"max_hessian_eigenval", np.max(eigenvals), prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"min_hessian_eigenval", np.min(eigenvals), prog_bar=False, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="test")

    def evaluate(self, batch, logits=None, stage=None):
        x, y = batch
        if logits is None:
            logits = self(x)
        loss = self.loss(logits, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}


def get_lnn_regression_model(in_dim, out_dim,
                             train_size=512,
                             activation_func=torch.nn.Identity(),
                             width=50,
                             N=2):
    model = LitNNModel(lr=1e-4,
                       weight_decay=5e-5,
                       n_features=in_dim,
                       n_classes=out_dim,
                       activation_func=activation_func,
                       hidden_layers_list=[width] * N,
                       loss_func=torch.nn.MSELoss(),
                       train_size=train_size,
                       )
    model.to(torch.float64)  # to support the (california houses) regression data

    trainer = Trainer(
        max_epochs=200,
        accelerator="auto",
        devices="auto",
        logger=[CSVLogger(save_dir="logs/", flush_logs_every_n_steps=100),
                TensorBoardLogger("tb_logs", name=f"model-{model._get_name()}")],
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
        ],
    )

    return model, trainer
