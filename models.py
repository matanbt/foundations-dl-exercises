from torch import nn

"""
Module for Torch models, dedicated for CIFAR10 image classification task
"""

CIFAR10_FLATTENED_IMG_DIM = 32 * 32 * 3
CIFAR10_NUM_CLASSES = 10


class BaselineNN(nn.Module):

    def __init__(self,
                 flattened_img_dim: int = CIFAR10_FLATTENED_IMG_DIM,
                 num_classes: int = CIFAR10_NUM_CLASSES,
                 ):
        super().__init__()
        hidden_layer_dim = 256
        self.model = nn.Sequential(
            nn.Flatten(),  # flattens the image
            nn.Linear(flattened_img_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, img):
        logits = self.model(img)

        return logits  # before softmax

