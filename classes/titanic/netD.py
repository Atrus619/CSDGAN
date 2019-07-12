import torch.nn as nn
import torch
from classes.NetUtils import NetUtils, CustomCatGANLayer
import torch.optim as optim


# Discriminator class
class netD(nn.Module, NetUtils):
    def __init__(self, device, H, out_dim, nc, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super().__init__()
        NetUtils.__init__(self)
        self.device = device
        self.loss_real = None
        self.loss_fake = None
        self.loss = None
        self.D_x = None
        self.D_G_z1 = None

        # Layers
        self.fc1 = nn.Linear(out_dim + nc, H, bias=True)
        # self.fc1_bn = nn.BatchNorm1d(H)
        self.output = nn.Linear(H, 1, bias=True)
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Sigmoid()

        # Loss and Optimizer
        self.loss_fn = nn.BCELoss()  # BCE Loss
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Record history of training
        self.init_layer_list()
        self.init_history()
        self.losses = []
        self.Avg_D_reals = []
        self.Avg_D_fakes = []

        # Initialize weights
        self.weights_init()

    def forward(self, row, labels):
        """
        :param row: Row of input data to discriminate on
        :param labels: Label embedding
        :return: Binary classification (sigmoid activation on a single unit hidden layer)
        """
        x = torch.cat([row, labels], 1)
        x = self.act(self.fc1(x))
        return self.m(self.output(x))

    def train_one_step_real(self, output, label):
        self.zero_grad()
        self.loss_real = self.loss_fn(output, label)
        self.loss_real.backward()
        self.D_x = output.mean().item()

    def train_one_step_fake(self, output, label):
        self.loss_fake = self.loss_fn(output, label)
        self.loss_fake.backward()
        self.D_G_z1 = output.mean().item()

    def combine_and_update_opt(self):
        self.loss = self.loss_real + self.loss_fake
        self.opt.step()

    def update_history(self):
        self.update_gnormz(2)
        self.update_wnormz(2)
        self.losses.append(self.loss.item())
        self.Avg_D_reals.append(self.D_x)
        self.Avg_D_fakes.append(self.D_G_z1)
