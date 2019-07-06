import torch.nn as nn
import torch
from models.NetUtils import NetUtils
import torch.optim as optim
import numpy as np


# Discriminator class
class CGAN_Discriminator(nn.Module, NetUtils):
    def __init__(self, nf, nc, num_channels, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super().__init__()
        NetUtils.__init__(self)

        self.loss_real = None
        self.loss_fake = None

        self.nc = nc
        self.nf = nf
        self.epoch = 0
        self.fc_labels_size = 128
        self.agg_size = 512

        # Convolutional layers
        # Image input size of num_channels x 28 x 28
        self.cn1 = nn.Conv2d(in_channels=num_channels, out_channels=self.nf, kernel_size=4, stride=2, padding=1, bias=True)
        self.cn1_bn = nn.BatchNorm2d(self.nf)
        # Intermediate size of nf x 14 x 14
        self.cn2 = nn.Conv2d(in_channels=self.nf, out_channels=self.nf * 2, kernel_size=4, stride=2, padding=1, bias=True)
        self.cn2_bn = nn.BatchNorm2d(self.nf * 2)
        # Intermediate size of nf*2 x 7 x 7

        # FC layers
        self.fc_labels = nn.Linear(in_features=self.nc, out_features=self.fc_labels_size, bias=True)
        self.fc_agg = nn.Linear(in_features=self.nf * 2 * 7 * 7 + self.fc_labels_size, out_features=self.agg_size, bias=True)
        self.fc_output = nn.Linear(in_features=self.agg_size, out_features=1, bias=True)

        # Activations
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Sigmoid()

        # Loss and Optimizer
        self.loss_fn = nn.BCELoss()  # BCE Loss combined with sigmoid for numeric stability
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Record history of training
        self.init_layer_list()
        self.init_history()

        self.D_x = []  # Per step
        self.Avg_D_reals = []  # D_x across epochs
        self.D_G_z1 = []  # Per step
        self.Avg_D_fakes = []  # Store D_G_z1 across epochs

        # Initialize weights
        self.weights_init()

    def forward(self, img, labels):
        """
        :param img: Input image of size 28 x 28
        :param labels: Label embedding
        :return: Binary classification (sigmoid activation on a single unit hidden layer)
        """
        x = self.act(self.cn1_bn(self.cn1(img)))
        x = self.act(self.cn2_bn(self.cn2(x)))
        x = x.view(-1, self.nf * 2 * 7 * 7)
        y = self.act(self.fc_labels(labels))
        agg = torch.cat((x, y), 1)
        agg = self.act(self.fc_agg(agg))
        return self.m(self.fc_output(agg))

    def train_one_step_real(self, output, label):
        self.zero_grad()
        self.loss_real = self.loss_fn(output, label)
        self.loss_real.backward()
        self.D_x.append(output.mean().item())

    def train_one_step_fake(self, output, label):
        self.loss_fake = self.loss_fn(output, label)
        self.loss_fake.backward()
        self.D_G_z1.append(output.mean().item())

    def combine_and_update_opt(self):
        self.loss.append(self.loss_real.item() + self.loss_fake.item())
        self.opt.step()
        self.store_weight_and_grad_norms()

    def next_epoch_discrim(self):
        """Discriminator specific actions"""
        self.Avg_D_reals.append(np.mean(self.D_x))  # Mean of means is not exact, but close enough for our purposes
        self.D_x = []

        self.Avg_D_fakes.append(np.mean(self.D_G_z1))
        self.D_G_z1 = []
