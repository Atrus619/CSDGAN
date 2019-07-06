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
from utils.utils import *


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

        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen

        self.fake_shuffle = True
        self.fake_num_workers = 6
        self.fake_data_set_size = fake_data_set_size
        self.fake_bs = fake_bs

        # Initialized through init_fake_gen method
        self.fake_train_set = None
        self.fake_train_gen = None
        self.fake_val_set = None
        self.fake_val_gen = None

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

    def train_gan(self, num_epochs, print_freq, eval_freq=None):
        """
        Primary method for training
        :param num_epochs: Desired number of epochs to train for
        :param print_freq: How freqently to print out training statistics (i.e., freq of 5 will result in information being printed every 5 epochs)
        :param eval_freq: How frequently to evaluate with netE. If None, no evaluation will occur. Evaluation takes a significant amount of time.
        """
        start_time = None
        for epoch in range(num_epochs):
            for x, y in self.train_gen:
                x, y = x.to(self.device), y.to(self.device)
                self.train_one_step(x, y)

            self.next_epoch()

            if self.epoch % print_freq == 0 or (self.epoch == num_epochs - 1):
                if start_time is not None:
                    print("Elapsed time since last eval: %ds" % (time.time() - start_time))
                start_time = time.time()

                self.print_progress(num_epochs)

                self.fixed_imgs.append(self.gen_fixed_img_grid())

            if eval_freq is not None:
                if self.epoch % eval_freq == 0 or (self.epoch == num_epochs - 1):
                    self.init_fake_gen()
                    self.test_model(train_gen=self.fake_train_gen, val_gen=self.fake_val_gen)
                    print("Epoch: %d\tEvaluator Score: %.4f" % (self.epoch+1, self.stored_acc[-1]))

    def train_one_step(self, x_train, y_train):
        """One full step of the CGAN training process"""
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

        # Update Generator
        labels.fill_(self.real_label)  # Reverse labels, fakes are real for generator cost
        # gen_fake_forward_pass = self.netD(x_train_fake, y_train).view(-1)
        gen_fake_forward_pass = self.netD(x_train_fake, garbage_y_train).view(-1)
        self.netG.train_one_step(gen_fake_forward_pass, labels)

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

    def print_progress(self, num_epochs):
        """Print metrics of interest"""
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (self.epoch+1, num_epochs, self.netD.losses[-1], self.netG.losses[-1], self.netD.Avg_D_reals[-1], self.netD.Avg_D_fakes[-1], self.netG.Avg_G_fakes[-1]))

        with torch.no_grad():
            # Generate sample of fake images to store for later
            self.fixed_imgs.append(self.gen_fixed_img_grid())

    def next_epoch(self):
        """Runs netG and netD methods to prepare for next epoch. Mostly saves histories and resets history collection objects."""
        self.epoch += 1

        self.netG.next_epoch()
        self.netG.next_epoch_gen()

        self.netD.next_epoch()
        self.netD.next_epoch_discrim()

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

    def show_img(self, label):
        """Generates a 28x28 image based on the desired class label index (integer 0-9)"""
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
        """ Plot describing progress over time of netE compared to an evaluation on real data"""
        pass

    def plot_training_plots(self, show=True, save=None):
        """
        Pulls together a plot of relevant training diagnostics for both netG and netD
        :param show: Whether to display the plot
        :param save: Whether to save the plot. If a value is entered, this is the path where the plot should be saved.
        """
        assert self.epoch > 0, "Model needs to be trained first"

        f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True)

        axes[0, 0].title.set_text("Generator and Discriminator Loss During Training")
        axes[0, 0].plot(self.netG.losses, label="G")
        axes[0, 0].plot(self.netD.losses, label="D")
        axes[0, 0].set_xlabel("iterations")
        axes[0, 0].set_ylabel("loss")
        axes[0, 0].legend()

        axes[0, 1].title.set_text("Average Discriminator Outputs During Training")
        axes[0, 1].plot(self.netD.Avg_D_reals, label="Real")
        axes[0, 1].plot(self.netD.Avg_D_fakes, label="Fake")
        axes[0, 1].plot(np.linspace(0, self.epoch, self.epoch), np.full(self.epoch, 0.5))
        axes[0, 1].set_xlabel("iterations")
        axes[0, 1].set_ylabel("proportion")
        axes[0, 1].legend()

        axes[1, 0].title.set_text('Gradient Norm History')
        axes[1, 0].plot(self.netG.gnorm_total_history, label="G")
        axes[1, 0].plot(self.netD.gnorm_total_history, label="D")
        axes[1, 0].set_xlabel("iterations")
        axes[1, 0].set_ylabel("norm")
        axes[1, 0].legend()

        axes[1, 1].title.set_text('Weight Norm History')
        axes[1, 1].plot(self.netG.wnorm_total_history, label="G")
        axes[1, 1].plot(self.netD.wnorm_total_history, label="D")
        axes[1, 1].set_xlabel("iterations")
        axes[1, 1].set_ylabel("norm")
        axes[1, 1].legend()

        st = f.suptitle("Training Diagnostic Plots", fontsize='x-large')
        f.tight_layout()
        st.set_y(0.96)
        f.subplots_adjust(top=0.9)

        if show:
            f.show()

        if save is not None:
            assert os.path.exists(save), "Check that the desired save path exists."
            safe_mkdir(save + '/training_plots')
            f.savefig(save + '/training_plots/training_plot.png')

    def load_netE(self, epoch):
        """Loads a previously stored netE (likely the one that performed the best)"""
        self.netE.load_state_dict(torch.load(self.netE_filepath + "/Epoch_" + epoch + "_Evaluator.pt"))
