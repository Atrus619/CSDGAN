import torch.nn as nn
import torch
from models.NetUtils import NetUtils
import torch.optim as optim


# Generator class
class CGAN_Generator(nn.Module, NetUtils):
    def __init__(self, nz, H, out_dim, nc, bs, lr, beta1, beta2):
        super().__init__()
        NetUtils.__init__(self)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.loss = None
        self.D_G_z2 = None
        self.fixed_noise = torch.randn(bs, nz, device=self.device)

        # Layers
        self.fc1 = nn.Linear(nz + nc, H, bias=True)
        # self.fc1_bn = nn.BatchNorm1d(H)
        self.output = nn.Linear(H, out_dim, bias=True)
        self.act = nn.LeakyReLU(0.2)

        # Loss and Optimizer
        self.loss_fn = nn.BCELoss()  # BCE Loss
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2))

        # Record history of training
        self.init_layer_list()
        self.init_hist()
        self.losses = []
        self.fixed_noise_outputs = []

        # Initialize weights
        self.weights_init()

    def forward(self, noise, labels):
        """
        Single dense hidden layer network.
        TODO: May want to adjust ReLU or add batch normalization later.
        :param noise: Random Noise vector Z
        :param labels: Label embedding
        :return: Row of data for iris data set (4 real values)
        """
        x = torch.cat([noise, labels], 1)
        x = self.act(self.fc1(x))
        return self.output(x)  # TODO: Make sure it is appropriate to not use an activation here

    def train_one_step(self, output, label):
        self.zero_grad()
        self.loss = self.loss_fn(output, label)
        self.loss.backward()
        self.D_G_z2 = output.mean().item()
        self.opt.step()

    def update_history(self):
        self.update_gnormz(2)
        self.update_wnormz(2)
        self.losses.append(self.loss.item())


# Discriminator class
class CGAN_Discriminator(nn.Module, NetUtils):
    def __init__(self, H, out_dim, nc, lr, beta1, beta2):
        super().__init__()
        NetUtils.__init__(self)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
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
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2))

        # Record history of training
        self.init_layer_list()
        self.init_hist()
        self.losses = []
        self.Avg_D_reals = []
        self.Avg_D_fakes = []

        # Initialize weights
        self.weights_init()

    def forward(self, row, labels):
        """
        Single dense hidden layer network.
        TODO: May want to adjust ReLU or add batch normalization later.
        :param row: Row of data from iris data set (4 real values)
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
