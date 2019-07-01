import torch.nn as nn
import torch
from models.NetUtils import NetUtils, CustomCatGANLayer
import torch.optim as optim


# This CGAN will be set up a bit differently in the hopes of being cleaner. I am going to enclose netG and netD into a higher level class titled CGAN.
class CGAN(nn.Module):
    def __init__(self, device, nz, num_channels, ngf, ndf, x_dim, nc, train_gen, val_gen, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super(CGAN, self).__init__()
        self.device = device
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.x_dim = x_dim
        self.nc = nc

        self.train_gen = train_gen
        self.val_gen = val_gen

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.wd = wd

        self.netG = CGAN_Generator(nz=self.nz, num_channels=num_channels, ngf=self.ngf, x_dim=self.x_dim, nc=self.nc, device=self.device,
                                   lr=self.lr, beta1=self.beta1, beta2=self.beta2, wd=self.wd).to(self.device)
        self.netD = CGAN_Discriminator(ndf=self.ndf, x_dim=self.x_dim, nc=self.nc,
                                       lr=self.lr, beta1=self.beta1, beta2=self.beta2, wd=self.wd).to(self.device)

        self.real_label = 1
        self.fake_label = 0

        self.epoch = 0

    def assemble_img_grid(self, imgs):
        pass

    def eval_fake_data(self):
        pass

    def train_one_step(self, x_train, y_train):
        bs = x_train.shape[0]

        # Update Discriminator, all real batch
        labels = torch.full((bs,), self.real_label, self.device)
        real_forward_pass = self.netD(x_train, y_train).view(-1)
        self.netD.train_one_step_real(real_forward_pass, labels)

        # Update Discriminator, all fake batch
        noise = torch.randn(bs, self.nz, device=self.device)
        x_train_fake = self.netG(noise, y_train)
        labels.fill_(self.fake_label)
        fake_forward_pass = self.netD(x_train_fake.detach(), y_train).view(-1)
        self.netD.train_one_step_fake(fake_forward_pass, labels)
        self.netD.combine_and_update_opt()
        self.netD.update_history()

        # Update Generator
        labels.fill_(self.real_label)  # Reverse labels, fakes are real for generator cost
        gen_fake_forward_pass = self.netD(x_train_fake, y_train).view(-1)
        self.netG.train_one_step(gen_fake_forward_pass, labels)
        self.netG.update_history()

    def eval_once(self, num_epochs):
        # Print metrics of interest
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (self.epoch+1, num_epochs, self.netD.loss.item(), self.netG.loss.item(), self.netD.D_x, self.netD.D_G_z1, self.netG.D_G_z2))

        with torch.no_grad():
            # Generate sample of fake images to store for later
            pass

            # Generate various levels of amounts of fake data and test how training compares


        # Update best scores and models


    def train_gan(self, num_epochs, eval_freq):
        for epoch in range(num_epochs):
            for x, y in self.train_gen:
                x, y = x.to(self.device), y.to(self.device)
                self.train_one_step(x, y)
            self.epoch += 1
            if self.epoch % eval_freq == 0 or (self.epoch == num_epochs-1):
                self.eval_once(num_epochs)


# Generator class
class CGAN_Generator(nn.Module, NetUtils):
    def __init__(self, nz, ngf, num_channels, x_dim, nc, device, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super(CGAN_Generator, self).__init__()
        self.loss = None
        self.D_G_z2 = None
        self.x_dim = x_dim
        self.fixed_noise = torch.randn(64, nz, device=device)
        # TODO: Fix the dimensions of the network using: https://pytorch.org/docs/stable/nn.html
        # Layers
        # Noise with category inputs
        self.ct1 = nn.ConvTranspose2d(in_channels=nz+nc, out_channels=ngf*4, kernel_size=4, stride=1, padding=0, bias=True)
        self.ct1_bn = nn.BatchNorm2d(ngf*4)
        # Intermediate size of (ngf*4) x 4 x 4
        self.ct2 = nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.ct2_bn = nn.BatchNorm2d(ngf*2)
        # Intermediate size of (ngf*2) x 8 x 8
        self.ct3 = nn.ConvTranspose2d(in_channels=ngf*2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=True)
        self.ct3_bn = nn.BatchNorm2d(ngf)
        # Intermediate size of (ngf*1) x 16 x 16
        self.output = nn.ConvTranspose2d(in_channels=ngf, out_channels=num_channels, kernel_size=4, stride=2, padding=1, bias=True)
        # Output size of num_channels x 32 x 32
        # Activations
        self.act = nn.LeakyReLU(0.2)
        self.sm = nn.Softmax(dim=-2)

        # Loss and Optimizer
        # TODO: Try Wasserstein distance instead of BCE Loss
        self.loss_fn = nn.BCELoss()  # BCE Loss
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Record history of training
        self.layer_list = [self._modules[x] for x in self._modules if self._modules[x].__class__.__name__.find('Linear') != -1]  # Return list of linear layers
        self.init_hist()
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
        return x.view(-1, self.x_dim)

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


# Discriminator class
class CGAN_Discriminator(nn.Module, NetUtils):
    def __init__(self, ndf, x_dim, nc, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super(CGAN_Discriminator, self).__init__()
        self.loss_real = None
        self.loss_fake = None
        self.loss = None
        self.D_x = None
        self.D_G_z1 = None

        # Layers
        self.fc1 = nn.Linear(x_dim + nc, H, bias=True)
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
