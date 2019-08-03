import shutil
import re
from torchviz import make_dot
import torch
import os
import utils.utils as uu
import matplotlib.pyplot as plt
import numpy as np


class CGANUtils:
    """Contains util methods to be inherited by both CGANs in this project"""

    def __init__(self):
        pass

    def init_paths(self):
        uu.safe_mkdir(self.path)
        stored_gen_path = os.path.join(self.path, "stored_generators")
        if os.path.exists(stored_gen_path):
            shutil.rmtree(stored_gen_path)
        uu.safe_mkdir(stored_gen_path)

    def train_one_step(self, x_train, y_train):
        """One full step of the CGAN training process"""
        bs = x_train.shape[0]
        self.netG.train()
        self.netD.train()
        y_train = y_train.float()  # Convert to float so that it can interact with float weights correctly

        # Update Discriminator, all real batch
        labels = (torch.rand(size=(bs,)) >= self.label_noise).type(torch.float32).to(self.device)
        real_forward_pass = self.netD(x_train, y_train).view(-1)
        self.netD.train_one_step_real(real_forward_pass, labels)

        # Update Discriminator, all fake batch
        noise = torch.randn(bs, self.nz, device=self.device)
        x_train_fake = self.netG(noise, y_train)
        labels = (torch.rand(size=(bs,)) <= self.label_noise).type(torch.float32).to(self.device)
        fake_forward_pass = self.netD(x_train_fake.detach(), y_train).view(-1)
        self.netD.train_one_step_fake(fake_forward_pass, labels)
        self.netD.combine_and_update_opt()

        for i in range(self.sched_netG):
            # Update Generator
            noise = torch.randn(bs, self.nz, device=self.device)
            x_train_fake = self.netG(noise, y_train)
            labels.fill_(self.real_label)  # Reverse labels, fakes are real for generator cost
            gen_fake_forward_pass = self.netD(x_train_fake, y_train).view(-1)
            self.netG.train_one_step(gen_fake_forward_pass, labels)

    def print_progress(self, total_epochs, run_id=None, logger=None):
        """Print metrics of interest"""
        statement = '[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (self.epoch, total_epochs, self.netD.losses[-1], self.netG.losses[-1],
                                                                                               self.netD.Avg_D_reals[-1], self.netD.Avg_D_fakes[-1], self.netG.Avg_G_fakes[-1])
        uu.train_log_print(run_id=run_id, logger=logger, statement=statement)

    def plot_training_plots(self, show=True, save=None):
        """
        Pull together a plot of relevant training diagnostics for both netG and netD
        :param show: Whether to display the plot
        :param save: Where to save the plots. If set to None default path is used. If false, not saved.
        """
        assert self.epoch > 0, "Model needs to be trained first"

        if save is None:
            save = self.path

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

        if save:
            assert os.path.exists(save), "Check that the desired save path exists."
            f.savefig(save + '/training_plot.png')

    def find_best_epoch(self):
        def parse_epoch(x):
            pattern = re.compile(r"[0-9]+")
            return int(re.findall(pattern=pattern, string=x)[0])

        gens = os.listdir(os.path.join(self.path, "stored_generators"))
        gens = sorted(gens, key=parse_epoch)
        return parse_epoch(gens[np.argmax(self.stored_acc) // len(self.test_ranges)])

    def load_netG(self, best=True, epoch=None):
        """Load a previously stored netG"""
        assert best or epoch is not None, "Either best arg must be True or epoch arg must not be None"

        if best:
            epoch = self.find_best_epoch()

        self.netG.load_state_dict(torch.load(self.path + "/stored_generators/Epoch_" + str(epoch) + "_Generator.pt"))

    def draw_architecture(self, net, show, save):
        """
        Utilize torchviz to print current graph to a pdf
        :param net: Network to draw graph for. One of netG, netD, or netE.
        :param show: Whether to show the graph. To visualize in jupyter notebooks, run the returned viz.
        :param save: Where to save the graph.
        """
        assert net in self.nets, "Invalid entry for net. Should be one of netG, netD, or netE"

        if save is None:
            save = self.path

        iterator = iter(self.train_gen)
        x, y = next(iterator)
        x, y = x.to(self.device), y.to(self.device).type(torch.float32)

        if net == self.netG:
            noise = torch.randn(x.shape[0], self.nz, device=self.device)
            viz = make_dot(net(noise, y), params=dict(net.named_parameters()))
        elif net == self.netD:
            viz = make_dot(net(x, y), params=dict(net.named_parameters()))
        else:
            viz = make_dot(net(x), params=dict(net.named_parameters()))

        title = net.name

        uu.safe_mkdir(save + "/architectures")
        viz.render(filename=save + "/architectures/" + title, view=show)

        return viz
