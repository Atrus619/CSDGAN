import torch.nn as nn
import torch
from classes.NetUtils import NetUtils, CustomCatGANLayer
import torch.optim as optim


# Generator class
class netG(nn.Module, NetUtils):
    def __init__(self, device, nz, H, out_dim, nc, x_dim, lr=2e-4, beta1=0.5, beta2=0.999, wd=0, cat_mask=None, le_dict=None):
        super().__init__()
        NetUtils.__init__(self)
        self.device = device
        self.loss = None
        self.D_G_z2 = None
        self.CCGL = CustomCatGANLayer(cat_mask=cat_mask, le_dict=le_dict)
        self.out_dim = out_dim

        # Layers
        self.fc1 = nn.Linear(nz + nc, H, bias=True)
        # self.fc1_bn = nn.BatchNorm1d(H)
        self.fc2 = nn.Linear(H, H, bias=True)
        self.fc3 = nn.Linear(H, H, bias=True)
        self.output = nn.Linear(H, self.out_dim, bias=True)
        self.act = nn.LeakyReLU(0.2)
        self.sm = nn.Softmax(dim=-2)

        # Loss and Optimizer
        # TODO: Try Wasserstein distance instead of BCE Loss
        self.loss_fn = nn.BCELoss()  # BCE Loss
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Record history of training
        self.init_layer_list()
        self.init_history()
        self.losses = []
        # self.fixed_noise_outputs = []

        # Initialize weights
        self.custom_weights_init()

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
        x = self.CCGL(x)
        return x.view(-1, self.out_dim)

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

    def custom_weights_init(self):
        for layer_name in self._modules:
            m = self._modules[layer_name]
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.uniform_(m.weight.data, -0.5, 0.5)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
