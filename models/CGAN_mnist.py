import torch.nn as nn
import torch
from models.NetUtils import NetUtils
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
# TODO: Write eval_fake_data method!!!

# This CGAN will be set up a bit differently in the hopes of being cleaner. I am going to enclose netG and netD into a higher level class titled CGAN.
class CGAN(nn.Module):
    def __init__(self, train_gen, val_gen, test_gen, device, nz, num_channels, ngf, ndf, x_dim, nc, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
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
        self.netD = CGAN_Discriminator(ndf=self.ndf, num_channels=num_channels, nc=self.nc,
                                       lr=self.lr, beta1=self.beta1, beta2=self.beta2, wd=self.wd).to(self.device)
        self.netE = CGAN_Evaluator(train_gen=train_gen, val_gen=val_gen, test_gen=test_gen, device=self.device, num_channels=num_channels, nc=nc,
                                   lr=lr, beta1=beta1, beta2=beta2, wd=wd)

        self.real_label = 1
        self.fake_label = 0

        self.epoch = 0

        self.fixed_imgs = []
        self.stored_models = []
        self.stored_scores = []

    def gen_fixed_img_grid(self):
        fixed_imgs = self.netG(self.netG.fixed_noise, self.netG.fixed_labels)
        return vutils.make_grid(tensor=fixed_imgs, nrow=self.nc, normalize=True).detach().cpu()

    def display_fixed_imgs(self, index):
        plt.imshow(np.transpose(self.fixed_imgs[index], (1, 2, 0)))

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
            self.fixed_imgs.append(self.gen_fixed_img_grid())
            # Print out current fixed fake images to monitor training progress
            self.display_fixed_imgs(-1)
            # Generate various levels of amounts of fake data and test how training compares
            self.eval_fake_data()

        # Update best scores and models
        pass

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
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        # 10x10, 10 examples of each of the 10 labels
        self.fixed_count_per_label = 10
        self.fixed_noise = torch.randn(100, nz, device=device)
        self.fixed_labels = self.init_fixed_labels().to(device)

        # Layers
        # Noise with one-hot encoded category conditional inputs
        self.fc1 = nn.Linear(in_features=self.nz + self.nc, out_features=7*7*ngf*2, bias=True)
        # Intermediate size of (ngf*2) x 7 x 7
        self.ct1 = nn.ConvTranspose2d(in_channels=self.ngf*2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1, bias=True)
        self.ct1_bn = nn.BatchNorm2d(ngf)
        # Intermediate size of (ngf*1) x 14 x 14
        self.output = nn.ConvTranspose2d(in_channels=self.ngf, out_channels=num_channels, kernel_size=4, stride=2, padding=1, bias=True)
        # Output size of num_channels x 28 x 28
        # Activations
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Tanh()

        # Loss and Optimizer
        # TODO: Try Wasserstein distance instead of BCE Loss
        self.loss_fn = nn.BCELoss()  # BCE Loss
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Record history of training
        self.layer_list = [self._modules[x] for x in self._modules if self._modules[x].__class__.__name__.find('ConvTranspose2d') != -1]  # Return list of linear layers
        self.init_hist()
        self.losses = []

        # Initialize weights
        self.custom_weights_init()

    def init_fixed_labels(self):
        tmp = torch.empty((self.nc * self.fixed_count_per_label, 1), dtype=torch.int64)
        for i in range(self.nc):
            tmp[i*self.fixed_count_per_label:((i+1)*self.fixed_count_per_label), ] = torch.full((self.fixed_count_per_label, 1), i)
        fixed_labels = torch.zeros(self.nc * self.fixed_count_per_label, self.nc)
        fixed_labels.scatter_(1, tmp, 1)
        return fixed_labels

    def forward(self, noise, labels):
        """
        Single dense hidden layer network.
        :param noise: Random Noise vector Z
        :param labels: Label embedding
        :return: MNIST img with values squashed by tanh to be between -1 and 1
        """
        x = torch.cat([noise, labels], 1)
        x = self.act(self.fc1(x))
        x = x.reshape(-1, self.ngf*2, 7, 7)
        x = self.act(self.ct1_bn(self.ct1(x)))
        return self.m(self.output(x))

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
    def __init__(self, ndf, nc, num_channels, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super(CGAN_Discriminator, self).__init__()
        self.loss_real = None
        self.loss_fake = None
        self.loss = None
        self.D_x = None
        self.D_G_z1 = None
        self.nc = nc
        self.ndf = ndf
        self.fc_labels_size = 128
        self.agg_size = 512

        # Convolutional layers
        # Image input size of num_channels x 28 x 28
        self.cn1 = nn.Conv2d(in_channels=num_channels, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=True)
        self.cn1_bn = nn.BatchNorm2d(self.ndf)
        # Intermediate size of ndf x 14 x 14
        self.cn2 = nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.cn2_bn = nn.BatchNorm2d(self.ndf*2)
        # Intermediate size of ndf*2 x 7 x 7

        # FC layers
        self.fc_labels = nn.Linear(in_features=self.nc, out_features=self.fc_labels_size, bias=True)
        self.fc_agg = nn.Linear(in_features=self.ndf*2*7*7+self.fc_labels_size, out_features=self.agg_size, bias=True)
        self.fc_output = nn.Linear(in_features=self.agg_size, out_features=1, bias=True)

        # Activations
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Sigmoid()

        # Loss and Optimizer
        self.loss_fn = nn.BCELoss()  # BCE Loss
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
        x = x.view(-1, self.ndf*2*7*7)
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


# Evaluator class
class CGAN_Evaluator(nn.Module):
    def __init__(self, train_gen, val_gen, test_gen, device, num_channels, nc, lr, beta1, beta2, wd):
        super(CGAN_Evaluator, self).__init__()

        # Generators
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen

        self.device = device

        # Layers
        self.cn1 = nn.Conv2d(in_channels=num_channels, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.cn1_bn = nn.BatchNorm2d(10)
        self.cn2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.cn2_bn = nn.BatchNorm2d(20)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc1 = nn.Linear(in_features=14*14*20, out_features=64)
        self.output = nn.Linear(in_features=64, out_features=nc)

        # Activations
        self.do2d = nn.Dropout2d(0.2)
        self.do1d = nn.Dropout(0.2)
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Softmax()

        # Loss and Optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Initialize weights
        self.weights_init()

        # Record history
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        x = self.do2d(self.act(self.cn1_bn(self.cn1(x))))
        x = self.do2d(self.act(self.cn2_bn(self.cn2(x))))
        x = self.mp(x)
        x = x.view(-1, 14*14*20)
        x = self.do1d(self.act(self.fc1(x)))
        return self.m(self.output(x))

    def process_batch(self, x, labels):
        forward = self.forward(x)
        loss = self.loss_fn(forward, labels)
        return loss

    def train_one_epoch(self):
        self.train()
        train_loss = 0
        for batch, labels in self.train_gen:
            batch, labels = batch.to(self.device), labels.to(self.device)
            self.zero_grad()
            train_loss += self.process_batch(batch, labels)
        return train_loss / len(self.train_gen)

    def eval_once(self, gen):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, labels in gen:
                batch, labels = batch.to(self.device), labels.to(self.device)
                val_loss += self.process_batch(batch, labels)
        return val_loss / len(gen)

    def train_evaluator(self, num_epochs, eval_freq, es=None):
        for epoch in range(num_epochs):
            self.train_losses.append(self.train_one_epoch())

            if epoch % eval_freq == 0 or (epoch == num_epochs-1):
                self.val_losses.append(self.eval_once(self.val_gen))

                if es:
                    if np.argmax(self.val_losses) < epoch - es:
                        return True
        return True
