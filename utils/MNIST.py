import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def convert_y_to_one_hot(y):
    """Converts a tensor of labels to a one_hot encoded version"""
    new_y = torch.zeros([len(y), 10], dtype=torch.uint8, device='cpu')
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
