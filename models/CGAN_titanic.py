import torch.nn as nn
import torch
from models.NetUtils import NetUtils
import torch.optim as optim


# Generator class
class CGAN_Generator(nn.Module, NetUtils):
    def __init__(self, device, nz, H, out_dim, nc, bs, lr=2e-4, beta1=0.5, beta2=0.999, wd=0, cat_mask=None):
        super(CGAN_Generator, self).__init__()
        self.device = device
        self.loss = None
        self.D_G_z2 = None
        self.fixed_noise = torch.randn(bs, nz, device=self.device)
        self.scaler = scaler

        # Masks
        self.cat = torch.Tensor(cat_mask).nonzero()
        self.cont = torch.Tensor(~cat_mask).nonzero()

        # Layers
        self.fc1 = nn.Linear(nz + nc, H, bias=True)
        # self.fc1_bn = nn.BatchNorm1d(H)
        self.fc2 = nn.Linear(H, H, bias=True)
        self.fc3 = nn.Linear(H, H, bias=True)
        self.output = nn.Linear(H, out_dim, bias=True)
        self.act = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

        # Loss and Optimizer
        # TODO: Try Wasserstein distance instead of BCE Loss
        self.loss_fn = nn.BCELoss()  # BCE Loss
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Record history of training
        self.layer_list = [self._modules[x] for x in self._modules if self._modules[x].__class__.__name__.find('Linear') != -1]  # Return list of linear layers
        self.init_hist()
        self.losses = []
        self.fixed_noise_outputs = []

        # Initialize weights
        self.weights_init()

    def forward(self, noise, labels):
        """
        Single dense hidden layer network.
        :param noise: Random Noise vector Z
        :param labels: Label embedding
        :return: Row of data for iris data set (4 real values)
        """
        x = torch.cat([noise, labels], 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.output(x)
        return self.output_processing(x)

    def output_processing(self, input_layer):
        """
        TODO: Softmax for each categorical variable - https://medium.com/jungle-book/towards-data-set-augmentation-with-gans-9dd64e9628e6
        :param input_layer: fully connected input layer with size out_dim
        :return: output of forward pass
        """
        cont = input_layer[:, self.cont].squeeze()
        cont = self.scaler.inverse_transform(cont)

        cat = input_layer[:, self.cat].squeeze()
        cat = self.sig(cat)

        halfway = torch.cat([cont, cat], 1)

        ints = halfway[:, self.int_mask].squeeze()
        ints = torch.round(ints)

        non_ints = halfway[:, self.non_int_mask].squeeze()

        return torch.cat([non_ints, ints], 1)

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
    def __init__(self, device, H, out_dim, nc, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super(CGAN_Discriminator, self).__init__()
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
        self.layer_list = [self._modules[x] for x in self._modules if self._modules[x].__class__.__name__.find('Linear') != -1]  # Return list of linear layers
        self.init_hist()
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
