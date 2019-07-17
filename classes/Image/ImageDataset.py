# TODO:
"""
1. Clean up files in classes
2. Adjust prototypes/reports
3. Test again
4. FashionMNIST
5. Add config printing utility
6. Begin productionizing
7. Build ImageDataset file
8. Table of results for README
"""

from torch.utils import data
from utils.MNIST import *
import torch
import random


class ImageDataset(data.Dataset):
    """Accepts input from img_dataset_preprocesser method"""
    def __init__(self, x, y):
        if x.dim() == 3:
            x = x.reshape(-1, 1, x.shape[1], x.shape[2])

        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class GeneratedImageDataset(data.Dataset):
    def __init__(self, netG, size, nz, nc, bs, ohe, device, stratify=None):
        self.netG = netG

        self.size = size
        self.nz = nz
        self.nc = nc
        self.bs = bs

        self.ohe = ohe

        self.device = device

        self.y = self.gen_labels(stratify=stratify)

        self.x = self.gen_data()

    def gen_labels(self, stratify=None):
        """
        Generate labels for generating fake data
        :param stratify: How to proportion out the labels. If None, a straight average is used.
        :return: Tuple of one hot encoded labels and the labels themselves
        """
        # Generate array of counts per label to be generated
        if stratify is None:
            stratify = [1 / self.nc for i in range(self.nc)]
        counts = np.round(np.dot(stratify, self.size), decimals=0).astype('int')
        while np.sum(counts) != self.size:
            if np.sum(counts) > self.size:
                counts[random.choice(range(self.nc))] -= 1
            else:
                counts[random.choice(range(self.nc))] += 1
        # Generate array of label names
        labels = np.empty(self.size)
        current_index = 0
        for i in range(self.nc):
            labels[current_index:(current_index + counts[i])] = i
            current_index += counts[i]
        # One hot encode labels
        arr = self.ohe.transform(labels)
        # Convert to tensor
        return torch.from_numpy(arr)

    def gen_data(self):
        """Generate fake training data examples for netE. Requires prior run of gen_labels"""
        x = torch.empty((self.size, 1, 28, 28), dtype=torch.float32, device='cpu')
        self.netG.eval()
        num_batches = self.size // self.bs

        with torch.no_grad():
            for i in range(num_batches):
                noise = torch.randn(self.bs, self.nz, device=self.device)
                y_batch = self.y[i*self.bs:(i+1)*self.bs].to(self.device)
                self.x[i*self.bs:(i+1)*self.bs] = self.netG(noise, y_batch).to('cpu')
            if self.size > (self.bs * num_batches):  # Fill in remaining spots
                remaining = self.size - (self.bs * num_batches)
                noise = torch.randn(remaining, self.nz, device=self.device)
                y_batch = self.y[(self.bs*num_batches):].to(self.device)
                self.x[(self.bs*num_batches):] = self.netG(noise, y_batch).to('cpu')

        return x

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index], self.y[index]
