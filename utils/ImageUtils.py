from utils.utils import train_val_test_split, encode_y
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def img_dataset_preprocesser(x, y, splits, seed=None):
    y, le, ohe = encode_y(y)

    x = x.astype('float32')

    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x, y, splits=splits, random_state=seed)

    train_max = x_train.max()
    train_min = x_train.min()

    def min_max_scaler(tensor, max, min):
        return (tensor - min) / (max - min)

    x_train = min_max_scaler(tensor=x_train, max=train_max, min=train_min)
    x_val = min_max_scaler(tensor=x_val, max=train_max, min=train_min)
    x_test = min_max_scaler(tensor=x_test, max=train_max, min=train_min)

    return x_train, y_train, x_val, y_val, x_test, y_test, le, ohe


def convert_y_to_one_hot(y, nc):
    """Converts a tensor of labels to a one_hot encoded version"""
    new_y = torch.zeros([len(y), nc], dtype=torch.uint8, device='cpu')
    y = y.view(-1, 1)
    new_y.scatter_(1, y, 1)
    return new_y


def show_real_grid(x_train, y_train, nc=10):
    """Generates a grid of images on real data. Randomly selects examples from provided args."""
    pre_grid = torch.empty((nc*10, 1, 28, 28))
    for i in range(nc):
        # Collect 10 random samples of each label
        locs = np.random.choice(np.where(y_train == i)[0], 10, replace=False)
        pre_grid[nc*i:(nc*(i+1)), 0, :] = x_train[locs]

    grid = vutils.make_grid(tensor=pre_grid, nrow=10, normalize=True)

    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
