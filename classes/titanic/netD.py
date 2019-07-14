import torch.nn as nn
import torch
from classes.NetUtils import NetUtils, GaussianNoise
import torch.optim as optim
import numpy as np


# Discriminator class
class netD(nn.Module, NetUtils):
    def __init__(self, device, H, out_dim, nc, noise, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super().__init__()
        NetUtils.__init__(self)

        self.device = device

        self.loss_real = None
        self.loss_fake = None

        self.noise = GaussianNoise(device=self.device, sigma=noise)

        # Layers
        self.fc1 = nn.Linear(out_dim + nc, H, bias=True)
        self.output = nn.Linear(H, 1, bias=True)
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Sigmoid()

        # Loss and Optimizer
        self.loss_fn = nn.BCELoss()  # BCE Loss
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Record history of training
        self.init_layer_list()
        self.init_history()
        self.update_hist_list()

        self.D_x = []  # Per step
        self.Avg_D_reals = []  # D_x across epochs
        self.D_G_z1 = []  # Per step
        self.Avg_D_fakes = []  # Store D_G_z1 across epochs

        # Initialize weights
        self.weights_init()

    def forward(self, row, labels):
        """
        :param row: Row of input data to discriminate on
        :param labels: Label embedding
        :return: Binary classification (sigmoid activation on a single unit hidden layer)
        """
        row = self.noise(row)
        x = torch.cat([row, labels], 1)
        x = self.act(self.fc1(x))
        return self.m(self.output(x))

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
