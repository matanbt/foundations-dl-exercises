from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from sklearn.decomposition import PCA

# CONFIGURATION VARS: -----------------------------------------------------------
BATCH_SIZE = 64  # size of the loader batches
DATA_FRACTION = 0.1  # fraction of the data to actually load

print(f">> Initializing data with parameters: "
      f"BATCH_SIZE={BATCH_SIZE}, DATA_FRACTION={DATA_FRACTION}")
# Helpers: -----------------------------------------------------------
def sample_dataset(dataset: torch.utils.data.Dataset,
                   fraction: float = 0.1):
    """ Samples random `fraction` of the torch `dataset` """
    subset_indices = np.random.choice(len(dataset), int(fraction * len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, subset_indices)

    return subset


def dataset_to_np_array(sub_dataset: torch.utils.data.Subset) -> Tuple[np.array]:
    """ Given a CIFAR10 torch *Subset* `sub_dataset`, returns tuple of numpy array - (samples, targets) """

    x = sub_dataset.dataset.data[sub_dataset.indices]
    x = x / 255.0  # `.data` provides the raw data, thus we divide (again) with `255`
    x = x.reshape(x.shape[0], -1)
    y = np.array(sub_dataset.dataset.targets)[sub_dataset.indices]
    return x, y


# Importable CIFAR dataset objects: -----------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # converts to tensors and sets to range [0,1]
    # transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])

# Training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset = sample_dataset(trainset, fraction=DATA_FRACTION)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

# Test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testset = sample_dataset(testset, fraction=DATA_FRACTION)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# Verbal Classes
# idx_to_classes = testset.classes  # also equals to `trainset.classes`
# idx_to_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Numpy variations (can be used for Sklearn training)
trainset_x, trainset_y = dataset_to_np_array(trainset)
testset_x, testset_y = dataset_to_np_array(testset)

# dataset after PCA
PCA_obj = PCA(whiten=True, n_components=500)
PCA_obj.fit(trainset_x)

trainset_PCA = torch.utils.data.TensorDataset(torch.Tensor(PCA_obj.transform(trainset_x)),
                                              torch.Tensor(trainset_y).type(torch.LongTensor))
trainloader_PCA = torch.utils.data.DataLoader(trainset_PCA, batch_size=BATCH_SIZE, 
                                              shuffle=False, num_workers=2)

testset_PCA = torch.utils.data.TensorDataset(torch.Tensor(PCA_obj.transform(testset_x)),
                                              torch.Tensor(testset_y).type(torch.LongTensor))
testloader_PCA  = torch.utils.data.DataLoader(testset_PCA, batch_size=BATCH_SIZE, 
                                              shuffle=False, num_workers=2)

