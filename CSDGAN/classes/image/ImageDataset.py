# TODO:
"""
2. Adjust prototypes/reports
3. Test again
4. FashionMNIST
6. Begin productionizing
8. Table of results for README
9. Stretch Goal: Add Dropout?!?
10. Fix hard-coded values throughout
11. Stretch Goal: Add automatic augmentation to help ImageCGAN
12. Stretch Goal: Histogram GIF with static axes
"""

import utils.ImageUtils as IU

from torch.utils import data
import random
import torchvision.transforms as t
from torchvision.datasets.folder import ImageFolder
import os
import numpy as np
import torch


class ImageDataset(data.Dataset):
    """Accepts input from img_dataset_preprocesser method. Assumes data set in (batch, channel, height, width) format."""

    def __init__(self, x, y):

        og_x_dim = (x.shape[-2], x.shape[-1])

        # If image only has 1 channel, it may need to be reshaped
        if x.dim() == 3:
            x = x.view(-1, 1, og_x_dim[0], og_x_dim[1])

        # Crop image for advantageous dimensions
        h_best_crop, _, _ = IU.find_pow_2_arch(og_x_dim[0])
        w_best_crop, _, _ = IU.find_pow_2_arch(og_x_dim[1])

        transformer = t.Compose([
            t.ToPILImage(),
            t.CenterCrop((og_x_dim[0] - h_best_crop, og_x_dim[1] - w_best_crop)),
            t.ToTensor()
        ])

        for i in range(len(x)):
            x[i] = transformer(x[i])

        # Finalize data set
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class GeneratedImageDataset(data.Dataset):
    def __init__(self, netG, size, nz, nc, bs, ohe, device, x_dim, stratify=None):
        self.netG = netG

        self.size = size
        self.nz = nz
        self.nc = nc
        self.bs = bs
        self.x_dim = x_dim

        self.ohe = ohe

        self.device = device

        self.x, self.y = self.gen_data(stratify=stratify)

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
        arr = self.ohe.transform(labels.reshape(-1, 1))
        # Convert to tensor
        return torch.from_numpy(arr).type(dtype=torch.float32)

    def gen_data(self, stratify=None):
        """Generate fake training data examples for netE. Requires prior run of gen_labels"""
        y = self.gen_labels(stratify=stratify)
        x = torch.empty((self.size, self.nc, self.x_dim[0], self.x_dim[1]), dtype=torch.float32, device='cpu')
        self.netG.eval()
        num_batches = self.size // self.bs

        with torch.no_grad():
            for i in range(num_batches):
                noise = torch.randn(self.bs, self.nz, device=self.device)
                y_batch = y[i * self.bs:(i + 1) * self.bs].to(self.device)
                x[i * self.bs:(i + 1) * self.bs] = self.netG(noise, y_batch).to('cpu')
            if self.size > (self.bs * num_batches):  # Fill in remaining spots
                remaining = self.size - (self.bs * num_batches)
                noise = torch.randn(remaining, self.nz, device=self.device)
                y_batch = y[(self.bs * num_batches):].to(self.device)
                x[(self.bs * num_batches):] = self.netG(noise, y_batch).to('cpu')

        return x, y

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class ImageFolderWithPaths(ImageFolder):
    """
    Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        img_id = os.path.basename(self.imgs[index][0])
        # make a new tuple that includes original and the path
        tuple_with_id = (original_tuple + (img_id,))
        return tuple_with_id


class OnlineGeneratedImageDataset(data.Dataset):
    def __init__(self, netG, size, nz, nc, bs, ohe, device, x_dim, stratify=None):
        self.netG = netG

        self.size = size
        self.nz = nz
        self.nc = nc
        self.bs = bs
        self.x_dim = x_dim

        self.ohe = ohe

        self.device = device

        self.full_y = self.gen_labels(stratify=stratify)
        self.internal_counter = 0
        self.x, self.y = None, None

        self.batches_per_epoch = self.size // self.bs + 1

    def gen_labels(self, stratify=None):
        """
        Generate labels for generating fake data
        :param stratify: How to proportion out the labels. If None, a straight average is used.
        :return: One hot encoded labels for the entire generator (shuffled)
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
        arr = self.ohe.transform(labels.reshape(-1, 1))
        # Shuffle order of labels
        np.random.shuffle(arr)
        # Convert to tensor
        return torch.from_numpy(arr).type(dtype=torch.float32)

    def gen_data(self, start, stop):
        """Generate fake training data examples for netE. Requires prior run of gen_labels"""
        stop = min(self.size, stop)  # Cap at length of data set
        y = self.full_y[start:stop].to(self.device)
        self.netG.eval()

        with torch.no_grad():
            noise = torch.randn(y.shape[0], self.nz, device=self.device)
            x = self.netG(noise, y).to('cpu')
        y = y.to('cpu')

        return x, y

    def next_batch(self):
        start, stop = self.internal_counter, self.internal_counter + self.bs
        self.internal_counter += self.bs
        return self.gen_data(start=start, stop=stop)

    def next_epoch(self):
        self.internal_counter = 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index % self.bs], self.y[index % self.bs]
