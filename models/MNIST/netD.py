import torch.nn as nn
import torch
from models.NetUtils import NetUtils
import torch.optim as optim


# Discriminator class
class CGAN_Discriminator(nn.Module, NetUtils):
    def __init__(self, nf, nc, num_channels, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super(CGAN_Discriminator, self).__init__()
        self.loss_real = None
        self.loss_fake = None
        self.loss = None
        self.D_x = None
        self.D_G_z1 = None
        self.nc = nc
        self.nf = nf
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
        self.layer_list = [self._modules[x] for x in self._modules if self._modules[x].__class__.__name__.find('Conv2d') != -1]  # Return list of linear layers
        self.init_hist()
        self.losses = []
        self.Avg_D_reals = []
        self.Avg_D_fakes = []

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
