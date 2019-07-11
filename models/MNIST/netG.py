import torch.nn as nn
import torch
from models.NetUtils import NetUtils
import torch.optim as optim
import numpy as np


# Generator class
class CGAN_Generator(nn.Module, NetUtils):
    def __init__(self, nz, nf, num_channels, x_dim, nc, device, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super().__init__()
        NetUtils.__init__(self)
        self.device = device

        self.num_channels = num_channels
        self.x_dim = x_dim
        self.nc = nc
        self.nz = nz
        self.nf = nf
        self.epoch = 0

        self.fixed_count_per_label = 10
        self.fixed_noise = torch.randn(self.fixed_count_per_label * nc, nz, device=self.device)  # 10x10, 10 examples of each of the 10 labels
        self.fixed_labels = self.init_fixed_labels().to(self.device)

        # Layers
        # Noise with one-hot encoded category conditional inputs
        self.ct1 = nn.ConvTranspose2d(in_channels=self.nz + self.nc, out_channels=self.nf*2, kernel_size=7, stride=1, padding=0, bias=True)
        self.ct1_bn = nn.BatchNorm2d(self.nf*2)
        # Intermediate size of (nf*2) x 7 x 7
        self.ct2 = nn.ConvTranspose2d(in_channels=self.nf*2, out_channels=self.nf, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
        self.ct2_bn = nn.BatchNorm2d(self.nf)
        # Intermediate size of (nf*1) x 14 x 14
        self.output = nn.ConvTranspose2d(in_channels=self.nf, out_channels=self.num_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
        # Output size of num_channels x 28 x 28
        # Activations
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Sigmoid()

        # Loss and Optimizer
        # TODO: Try Wasserstein distance instead of BCE Loss
        self.loss_fn = nn.BCELoss()  # BCE Loss
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Initialize weights
        self.weights_init()

        # Record history of training
        self.init_layer_list()
        self.init_history()
        self.update_hist_list()

        self.D_G_z2 = []  # Per step
        self.Avg_G_fakes = []  # Store D_G_z2 across epochs

    def init_fixed_labels(self):
        tmp = torch.empty((self.nc * self.fixed_count_per_label, 1), dtype=torch.int64)
        for i in range(self.nc):
            tmp[i * self.fixed_count_per_label:((i + 1) * self.fixed_count_per_label), ] = torch.full((self.fixed_count_per_label, 1), i)
        fixed_labels = torch.zeros(self.nc * self.fixed_count_per_label, self.nc)
        fixed_labels.scatter_(1, tmp, 1)
        return fixed_labels

    def forward(self, noise, labels):
        """
        Single dense hidden layer network.
        :param noise: Random Noise vector Z
        :param labels: Label embedding of labels
        :return: MNIST img with values squashed by sigmoid to be between 0 and 1
        """
        x = torch.cat([noise, labels], -1).view(-1, self.nz + self.nc, 1, 1)
        x = self.act(self.ct1_bn(self.ct1(x)))
        x = self.act(self.ct2_bn(self.ct2(x)))
        return self.m(self.output(x))

    def train_one_step(self, output, label):
        self.zero_grad()
        loss_tmp = self.loss_fn(output, label)
        loss_tmp.backward()
        self.loss.append(loss_tmp.item())
        self.D_G_z2.append(output.mean().item())
        self.opt.step()

        self.store_weight_and_grad_norms()

    def next_epoch_gen(self):
        """Generator specific actions"""
        self.Avg_G_fakes.append(np.mean(self.D_G_z2))
        self.D_G_z2 = []
