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
                 hidden_layer_dim: int = 256,
                 hidden_layers_count: int = 1,  # hidden layers count, must be > 1
                 p_dropout: float = 0,  #  probability of an element to be zeroed
                 ):
        super().__init__()

        layers = []

        # flattens the image
        layers.append(nn.Flatten())
        # hidden layers:
        layers += [
            nn.Linear(flattened_img_dim, hidden_layer_dim),
            nn.Dropout(p=p_dropout),
            nn.ReLU(),
        ]
        for _ in range(hidden_layers_count - 1):
            layers += [
                nn.Linear(hidden_layer_dim, hidden_layer_dim),
                nn.Dropout(p=p_dropout),
                nn.ReLU(),
            ]
        # project to output:
        layers.append(nn.Linear(hidden_layer_dim, num_classes))
        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, img):
        logits = self.model(img)

        return logits  # before softmax

