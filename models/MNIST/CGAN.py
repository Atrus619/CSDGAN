import torch.nn as nn
import matplotlib.animation as animation
import numpy as np
from PytorchDatasets.MNIST_Dataset import Fake_MNIST_Dataset
from torch.utils import data
from models.MNIST.netD import CGAN_Discriminator
from models.MNIST.netG import CGAN_Generator
from models.MNIST.netE import CGAN_Evaluator
from utils.MNIST import *
import time


# This CGAN will be set up a bit differently in the hopes of being cleaner. I am going to enclose netG and netD into a higher level class titled CGAN.
class CGAN(nn.Module):
    def __init__(self, train_gen, val_gen, test_gen, device, x_dim, nc, nz, num_channels, netE_filepath,
                 netG_nf, netG_lr, netG_beta1, netG_beta2, netG_wd,
                 netD_nf, netD_lr, netD_beta1, netD_beta2, netD_wd,
                 netE_lr, netE_beta1, netE_beta2, netE_wd,
                 fake_data_set_size, fake_bs,
                 eval_num_epochs, early_stopping_patience):
        # Inherit nn.Module initialization
        super(CGAN, self).__init__()
        #
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen

        self.fake_shuffle = True
        self.fake_num_workers = 6
        self.fake_data_set_size = fake_data_set_size
        self.fake_bs = fake_bs
        self.fake_train_set = None  # Initialized through init_fake_gen method
        self.fake_train_gen = None  # Initialized through init_fake_gen method
        self.fake_val_set = None  # Initialized through init_fake_gen method
        self.fake_val_gen = None  # Initialized through init_fake_gen method

        self.device = device
        self.x_dim = x_dim
        self.nc = nc
        self.nz = nz
        self.num_channels = num_channels
        self.netE_filepath = netE_filepath

        self.real_label = 1
        self.fake_label = 0

        self.epoch = 0

        self.fixed_imgs = []
        self.stored_loss = []
        self.stored_acc = []

        # Instantiate sub-nets
        self.netG = CGAN_Generator(nz=self.nz, num_channels=self.num_channels, nf=netG_nf, x_dim=self.x_dim, nc=self.nc, device=self.device,
                                   lr=netG_lr, beta1=netG_beta1, beta2=netG_beta2, wd=netG_wd).to(self.device)
        self.netD = CGAN_Discriminator(nf=netD_nf, num_channels=self.num_channels, nc=self.nc,
                                       lr=netD_lr, beta1=netD_beta1, beta2=netD_beta2, wd=netD_wd).to(self.device)

        self.netE_params = {'lr': netE_lr, 'beta1': netE_beta1, 'beta2': netE_beta2, 'wd': netE_wd}
        self.netE = None  # Initialized through init_evaluator method
        self.eval_num_epochs = eval_num_epochs
        self.early_stopping_patience = early_stopping_patience

    def train_gan(self, num_epochs, print_freq, use_netE=False):
        start_time = None
        for epoch in range(num_epochs):
            for x, y in self.train_gen:
                x, y = x.to(self.device), y.to(self.device)
                self.train_one_step(x, y)
            if self.epoch % print_freq == 0 or (self.epoch == num_epochs - 1):
                if start_time is not None:
                    print("Elapsed time since last eval: %ds" % (time.time() - start_time))
                start_time = time.time()
                self.print_progress(num_epochs)

                if use_netE:
                    self.init_fake_gen()
                    self.test_model(train_gen=self.fake_train_gen, val_gen=self.fake_val_gen)
                    print("Evaluator Score: %.4f" % (self.stored_acc[-1]))

                self.fixed_imgs.append(self.gen_fixed_img_grid())
            self.epoch += 1

    def train_one_step(self, x_train, y_train):
        bs = x_train.shape[0]
        self.netG.train()
        self.netD.train()
        y_train = y_train.float()  # Convert to float so that it can interact with float weights correctly

        garbage_y_train = convert_y_to_one_hot(torch.from_numpy(np.random.randint(0, 9, len(y_train)))).to(self.device).type(torch.float32)
        # import pdb; pdb.set_trace()
        # Update Discriminator, all real batch
        labels = torch.full(size=(bs,), fill_value=self.real_label, device=self.device)
        # real_forward_pass = self.netD(x_train, y_train).view(-1)
        real_forward_pass = self.netD(x_train, garbage_y_train).view(-1)
        self.netD.train_one_step_real(real_forward_pass, labels)

        # Update Discriminator, all fake batch
        noise = torch.randn(bs, self.nz, device=self.device)
        x_train_fake = self.netG(noise, y_train)
        labels.fill_(self.fake_label)
        # fake_forward_pass = self.netD(x_train_fake.detach(), y_train).view(-1)
        fake_forward_pass = self.netD(x_train_fake.detach(), garbage_y_train).view(-1)
        self.netD.train_one_step_fake(fake_forward_pass, labels)
        self.netD.combine_and_update_opt()
        self.netD.update_history()

        # Update Generator
        labels.fill_(self.real_label)  # Reverse labels, fakes are real for generator cost
        # gen_fake_forward_pass = self.netD(x_train_fake, y_train).view(-1)
        gen_fake_forward_pass = self.netD(x_train_fake, garbage_y_train).view(-1)
        self.netG.train_one_step(gen_fake_forward_pass, labels)
        self.netG.update_history()

    def test_model(self, train_gen, val_gen):
        """
        Train a CNN evaluator from scratch
        :param train_gen: Specified train_gen, can either be real training generator or a created one from netG
        :param val_gen: Same as above ^
        :param num_epochs: Number of epochs to train for
        :param es: Early-stopping, None by default
        :return: Best performance on test set
        """
        self.init_evaluator(train_gen, val_gen)
        self.netE.train_evaluator(num_epochs=self.eval_num_epochs, eval_freq=1, es=self.early_stopping_patience)
        torch.save(self.netE.state_dict, self.netE_filepath + "/Epoch_" + str(self.epoch) + "_Evaluator.pt")
        loss, acc = self.netE.eval_once(self.test_gen)
        self.stored_loss.append(loss.item())
        self.stored_acc.append(acc.item())

    def init_evaluator(self, train_gen, val_gen):
        """
        Initializes the netE sub-net. This is done as a separate method because we want to reinitialize netE each time we want to evaluate it.
        We can also evaluate on the original, real data by specifying these training generators.
        """
        self.netE = CGAN_Evaluator(train_gen=train_gen, val_gen=val_gen, test_gen=self.test_gen, device=self.device, num_channels=self.num_channels, nc=self.nc,
                                   **self.netE_params).to(self.device)

    def init_fake_gen(self):
        # Initializes fake training set and validation set to be same size
        self.fake_train_set = Fake_MNIST_Dataset(self.netG, self.fake_data_set_size, self.nz, self.nc, self.device)
        self.fake_train_gen = data.DataLoader(self.fake_train_set, batch_size=self.fake_bs, shuffle=self.fake_shuffle, num_workers=self.fake_num_workers)

        self.fake_val_set = Fake_MNIST_Dataset(self.netG, self.fake_data_set_size, self.nz, self.nc, self.device)
        self.fake_val_gen = data.DataLoader(self.fake_val_set, batch_size=self.fake_bs, shuffle=self.fake_shuffle, num_workers=self.fake_num_workers)

    def print_progress(self, num_epochs):
        # Print metrics of interest
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (self.epoch + 1, num_epochs, self.netD.loss.item(), self.netG.loss.item(), self.netD.D_x, self.netD.D_G_z1, self.netG.D_G_z2))

        with torch.no_grad():
            # Generate sample of fake images to store for later
            self.fixed_imgs.append(self.gen_fixed_img_grid())

    def show_img(self, label):
        # Generates a 28x28 image based on the desired class label index (integer 0-9)
        assert 0 <= label <= 9 and type(label) is int, "Make sure label is an integer between 0 and 9 (inclusive)."
        noise = torch.randn(1, self.nz, device=self.device)
        processed_label = torch.zeros([1, 10], dtype=torch.uint8, device='cpu')
        label = torch.full((1, 1), label, dtype=torch.int64)
        processed_label = processed_label.scatter(1, label, 1).float().to(self.device)
        output = self.netG(noise, processed_label).view(28, 28).detach().cpu().numpy()
        plt.imshow(output)
        plt.show()

    def gen_fixed_img_grid(self):
        """
        Produces a grid of generated images from netG's fixed noise vector. This can be used to visually track progress of the CGAN training.
        :return: Tensor of images
        """
        self.netG.eval()
        with torch.no_grad():
            fixed_imgs = self.netG(self.netG.fixed_noise, self.netG.fixed_labels)
        return vutils.make_grid(tensor=fixed_imgs, nrow=10, normalize=True).detach().cpu()

    def show_grid(self, index):
        """
        Prints a specified fixed image grid from the self.fixed_imgs list
        :param index: Evaluation index to display
        :return: Nothing. Displays the desired image instead.
        """
        if len(self.fixed_imgs) == 0:
            print("Model not yet trained.")
        else:
            fig = plt.figure(figsize=(8, 8))
            plt.axis('off')
            plt.imshow(np.transpose(self.fixed_imgs[index], (1, 2, 0)))
            plt.show()

    def show_video(self):
        """
        Produces a video demonstrating the CGAN's progress over evaluations.
        :return: Nothing. Displays the desired video instead.
        """
        if len(self.fixed_imgs) == 0:
            print("Model not yet trained.")
        else:
            fig = plt.figure(figsize=(8, 8))
            plt.axis('off')
            ims = [[plt.imshow(np.transpose(grid, (1, 2, 0)))] for grid in self.fixed_imgs]
            ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
            plt.show()

    def plot_progress(self):
        pass

    def load_netE(self, epoch):
        # Loads a previously stored netE (likely the one that performed the best)
        self.netE.load_state_dict(torch.load(self.netE_filepath + "/Epoch_" + epoch + "_Evaluator.pt"))
