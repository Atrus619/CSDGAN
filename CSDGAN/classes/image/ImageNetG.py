import utils.ImageUtils as IU
from CSDGAN.classes.NetUtils import NetUtils

import torch.optim as optim
from collections import OrderedDict
import torch
import numpy as np
import torch.nn as nn


# Generator class
class ImageNetG(nn.Module, NetUtils):
    def __init__(self, nz, nf, num_channels, path, x_dim, nc, device, grid_nrow, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super().__init__()
        NetUtils.__init__(self)
        self.name = "Generator"

        self.path = path
        self.device = device

        self.num_channels = num_channels
        self.x_dim = x_dim
        self.nc = nc
        self.nz = nz
        self.nf = nf
        self.epoch = 0

        self.grid_nrow = grid_nrow
        self.fixed_noise = torch.randn(self.grid_nrow * self.nc, self.nz, device=self.device)
        self.fixed_labels = self.init_fixed_labels().to(self.device)

        # Layers
        self.arch = OrderedDict()
        self.output = None  # Initialized in line below
        self.assemble_architecture(h=self.x_dim[0], w=self.x_dim[1])

        # Activations
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Sigmoid()

        # Loss and Optimizer
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
        tmp = torch.empty((self.grid_nrow * self.nc, 1), dtype=torch.int64)
        for i in range(self.nc):
            tmp[i * self.grid_nrow:((i + 1) * self.grid_nrow), ] = torch.full((self.grid_nrow, 1), i)
        fixed_labels = torch.zeros(self.nc * self.grid_nrow, self.nc)
        fixed_labels.scatter_(1, tmp, 1)
        return fixed_labels

    def forward(self, noise, labels):
        """
        Deep Convolutional Upsampling Network of Variable Image Size (on creation only)
        layer[0] = ConvTranspose2d
        layer[1] = BatchNorm2d
        :param noise: Random Noise vector Z (must be float!)
        :param labels: Label embedding of labels (must be float!)
        :return: Image of dimensions num_channels x cropped image size with values squashed by sigmoid to be between 0 and 1
        """
        x = torch.cat([noise, labels], -1).view(-1, self.nz + self.nc, 1, 1)
        for layer_name, layer in self.arch.items():
            x = self.act(layer[1](layer[0](x)))
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

    def assemble_architecture(self, h, w):
        """Fills in an ordered dictionaries with tuples, one for the layers and one for the corresponding batch norm layers"""
        h_best_crop, h_best_first, h_pow_2 = IU.find_pow_2_arch(h)
        w_best_crop, w_best_first, w_pow_2 = IU.find_pow_2_arch(w)
        assert (h_best_crop, w_best_crop) == (0, 0), "Crop not working properly"

        num_intermediate_upsample_layers = max(h_pow_2, w_pow_2) - 1  # Not counting the final layer
        self.arch['ct1'] = IU.first_block(h=h_best_first, w=w_best_first, in_channels=self.nz + self.nc,
                                          out_channels=self.nf * 2 ** num_intermediate_upsample_layers)
        self.add_module('ct1', self.arch['ct1'][0])
        self.add_module('ct1_bn', self.arch['ct1'][1])

        # Upsample by 2x until it is no longer necessary, then upsample by 1x
        h_rem, w_rem = self.x_dim[0] - h_best_crop, self.x_dim[1] - w_best_crop
        h_rem, w_rem = h_rem // h_best_first, w_rem // w_best_first
        for i in range(num_intermediate_upsample_layers):
            h_rem, w_rem, h_curr, w_curr = IU.update_h_w_curr(h_rem=h_rem, w_rem=w_rem)
            self.arch['ct' + str(i + 2)] = IU.ct2_upsample_block(h=h_curr, w=w_curr,
                                                                 in_channels=self.nf * 2 ** (num_intermediate_upsample_layers - i),
                                                                 out_channels=self.nf * 2 ** (num_intermediate_upsample_layers - (i + 1)))
            self.add_module('ct' + str(i + 2), self.arch['ct' + str(i + 2)][0])
            self.add_module('ct' + str(i + 2) + '_bn', self.arch['ct' + str(i + 2)][1])

        # Final layer
        h_rem, w_rem, h_curr, w_curr = IU.update_h_w_curr(h_rem=h_rem, w_rem=w_rem)
        self.output, __ = IU.ct2_upsample_block(h=h_curr, w=w_curr,
                                                in_channels=self.nf,
                                                out_channels=self.num_channels,
                                                add_op=(h_best_crop, w_best_crop))
