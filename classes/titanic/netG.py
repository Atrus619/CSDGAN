import torch.nn as nn
import torch
from classes.NetUtils import NetUtils, CustomCatGANLayer
import torch.optim as optim
import numpy as np


# Generator class
class netG(nn.Module, NetUtils):
    def __init__(self, device, nz, H, out_dim, nc, lr=2e-4, beta1=0.5, beta2=0.999, wd=0, cat_mask=None, le_dict=None):
        super().__init__()
        NetUtils.__init__(self)
        self.name = "Generator"

        self.device = device

        self.CCGL = CustomCatGANLayer(cat_mask=cat_mask, le_dict=le_dict)
        self.out_dim = out_dim

        # Layers
        self.fc1 = nn.Linear(nz + nc, H, bias=True)
        self.fc2 = nn.Linear(H, H, bias=True)
        self.fc3 = nn.Linear(H, H, bias=True)
        self.output = nn.Linear(H, self.out_dim, bias=True)
        self.act = nn.LeakyReLU(0.2)
        self.sm = nn.Softmax(dim=-2)

        # Loss and Optimizer
        self.loss_fn = nn.BCELoss()  # BCE Loss
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Initialize weights
        self.custom_weights_init()

        # Record history of training
        self.init_layer_list()
        self.init_history()
        self.update_hist_list()

        self.D_G_z2 = []  # Per step
        self.Avg_G_fakes = []  # Store D_G_z2 across epochs

    def forward(self, noise, labels):
        """
        Single dense hidden layer network.
        :param noise: Random Noise vector Z
        :param labels: Label embedding
        :return: Row of data
        """
        x = torch.cat([noise, labels], 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.output(x)
        x = self.CCGL(x)
        return x.view(-1, self.out_dim)

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

    def custom_weights_init(self):
        for layer_name in self._modules:
            m = self._modules[layer_name]
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.uniform_(m.weight.data, -0.5, 0.5)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
