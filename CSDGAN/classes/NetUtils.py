import imageio
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import utils.utils as uu
import os


class NetUtils:
    """Contains utils to be inherited by other nets in this project"""
    def __init__(self):
        self.epoch = 0
        self.streaming_weight_history = {}
        self.streaming_gradient_history = {}

        self.histogram_weight_history = {}
        self.histogram_gradient_history = {}

        self.gnorm_history = {}
        self.gnorm_total_history = []
        self.wnorm_history = {}
        self.wnorm_total_history = []

        self.layer_list = []
        self.layer_list_names = []

        self.loss = []  # List of loss per step
        self.losses = []  # List of loss per epoch

        self.norm_num = 2
        self.bins = 20  # Choice of bins=20 seems to look nice. Subject to change.

    def init_layer_list(self):
        """Initializes list of layers for tracking history"""
        nn_module_ignore_list = {'batchnorm', 'activation', 'loss', 'Noise', 'CustomCatGANLayer'}  # List of nn.modules to ignore when constructing layer_list
        self.layer_list = [x for x in self._modules.values() if not any(excl in str(type(x)) for excl in nn_module_ignore_list)]
        self.layer_list_names = [x for x in self._modules.keys() if not any(excl in str(type(self._modules[x])) for excl in nn_module_ignore_list)]

    def init_history(self):
        """Initializes objects for storing history based on layer_list"""
        for layer in self.layer_list:
            self.streaming_weight_history[layer] = {'weight': [], 'bias': []}
            self.streaming_gradient_history[layer] = {'weight': [], 'bias': []}

            self.histogram_weight_history[layer] = {'weight': [], 'bias': []}
            self.histogram_gradient_history[layer] = {'weight': [], 'bias': []}

            self.wnorm_history[layer] = {'weight': [], 'bias': []}
            self.gnorm_history[layer] = {'weight': [], 'bias': []}

    def next_epoch(self):
        """Resets internal storage of training history to stream next epoch"""
        self.epoch += 1

        self.losses.append(np.mean(self.loss))
        self.loss = []

        self.update_wnormz()
        self.update_gnormz()
        self.update_hist_list()

        for layer in self.layer_list:
            self.streaming_weight_history[layer] = {'weight': [], 'bias': []}
            self.streaming_gradient_history[layer] = {'weight': [], 'bias': []}

    def store_weight_and_grad_norms(self):
        """
        Appends training history for summarization and visualization later. Scales each norm by the number of elements.
        Should be ran once per step per subnet.
        """
        for layer in self.layer_list:
            self.streaming_weight_history[layer]['weight'].append(layer.weight.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.weight.numel())
            self.streaming_weight_history[layer]['bias'].append(layer.bias.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.bias.numel())

            self.streaming_gradient_history[layer]['weight'].append(layer.weight.grad.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.weight.grad.numel())
            self.streaming_gradient_history[layer]['bias'].append(layer.bias.grad.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.bias.grad.numel())

    def update_hist_list(self):
        """
        Updates the histogram history based on the weights at the end of an epoch.
        Should be ran once per epoch per subnet.
        """
        for layer in self.layer_list:
            self.histogram_weight_history[layer]['weight'].append(np.histogram(layer.weight.detach().cpu().numpy().reshape(-1), bins=self.bins))
            self.histogram_weight_history[layer]['bias'].append(np.histogram(layer.bias.detach().cpu().numpy().reshape(-1), bins=self.bins))

            if self.epoch == 0:  # Model is untrained; no gradients exist yet
                self.histogram_gradient_history[layer]['weight'].append(None)
                self.histogram_gradient_history[layer]['bias'].append(None)
            else:
                self.histogram_gradient_history[layer]['weight'].append(np.histogram(layer.weight.grad.detach().cpu().numpy().reshape(-1), bins=self.bins))
                self.histogram_gradient_history[layer]['bias'].append(np.histogram(layer.bias.grad.detach().cpu().numpy().reshape(-1), bins=self.bins))

    def update_wnormz(self):
        """
        Tracks history of desired norm of weights.
        Should be ran once per epoch per subnet.
        :param norm_num: 1 = l1 norm, 2 = l2 norm
        :return: list of norms of weights by layer, as well as overall weight norm
        """
        total_norm = 0
        for layer in self.wnorm_history:
            w_norm = np.linalg.norm(self.streaming_weight_history[layer]['weight'], self.norm_num)
            b_norm = np.linalg.norm(self.streaming_weight_history[layer]['bias'], self.norm_num)
            self.wnorm_history[layer]['weight'].append(w_norm)
            self.wnorm_history[layer]['bias'].append(b_norm)

            if self.norm_num == 1:
                total_norm += abs(w_norm) + abs(b_norm)
            else:
                total_norm += w_norm ** self.norm_num + b_norm ** self.norm_num

        total_norm = total_norm ** (1. / self.norm_num)
        self.wnorm_total_history.append(total_norm)

    def update_gnormz(self):
        """
        Calculates gradient norms by layer as well as overall. Scales each norm by the number of elements.
        Should be ran once per epoch per subnet.
        :param norm_num: 1 = l1 norm, 2 = l2 norm
        :return: list of gradient norms by layer, as well as overall gradient norm
        """
        total_norm = 0
        for layer in self.gnorm_history:
            w_norm = np.linalg.norm(self.streaming_gradient_history[layer]['weight'], self.norm_num) / len(self.streaming_gradient_history[layer]['weight'])
            b_norm = np.linalg.norm(self.streaming_gradient_history[layer]['bias'], self.norm_num) / len(self.streaming_gradient_history[layer]['bias'])

            self.gnorm_history[layer]['weight'].append(w_norm)
            self.gnorm_history[layer]['bias'].append(b_norm)
            if self.norm_num == 1:
                total_norm += abs(w_norm) + abs(b_norm)
            else:
                total_norm += w_norm**self.norm_num + b_norm**self.norm_num
        total_norm = total_norm**(1./self.norm_num) / len(self.gnorm_history)
        self.gnorm_total_history.append(total_norm)

    def weights_init(self):
        """
        Custom weights initialization for subnets
        Should only be run when first creating net. Will reset effects of training if run after training.
        """
        for layer_name in self._modules:
            m = self._modules[layer_name]
            classname = m.__class__.__name__

            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def plot_layer_scatters(self, figsize=(20, 10), show=True, save=None):
        """Plot weight and gradient norm history for each layer in layer_list across epochs"""
        assert self.epoch > 0, "Model needs to be trained first"

        if save is None:
            save = self.path

        f, axes = plt.subplots(len(self.layer_list), 4, figsize=figsize, sharex=True)

        axes[0, 0].title.set_text("Weight Norms")
        axes[0, 1].title.set_text("Weight Gradient Norms")
        axes[0, 2].title.set_text("Bias Norms")
        axes[0, 3].title.set_text("Bias Gradient Norms")

        for i in range(4):
            axes[len(self.layer_list) - 1, i].set_xlabel('epochs')

        for i, layer in enumerate(self.layer_list):
            axes[i, 0].set_ylabel(self.layer_list_names[i])
            axes[i, 0].plot(self.wnorm_history[layer]['weight'])
            axes[i, 1].plot(self.gnorm_history[layer]['weight'])
            axes[i, 2].plot(self.wnorm_history[layer]['bias'])
            axes[i, 3].plot(self.gnorm_history[layer]['bias'])

        sup = self.name + " Layer Weight and Gradient Norms"
        st = f.suptitle(sup, fontsize='x-large')
        f.tight_layout()
        st.set_y(0.96)
        f.subplots_adjust(top=0.9)

        if show:
            f.show()

        if save:
            assert os.path.exists(save), "Check that the desired save path exists."
            os.makedirs(os.path.join(save, 'layer_scatters'), exist_ok=True)
            f.savefig(os.path.join(save, 'layer_scatters', self.name + '_layer_scatters.png'))

    def plot_layer_hists(self, epoch=None, figsize=(20, 10), show=True, save=None):
        """Plots histograms of weight and gradients for each layer in layer_list at the desired epoch"""
        if epoch is None:
            epoch = self.epoch

        if save is None:
            save = self.path

        f, axes = plt.subplots(len(self.layer_list), 4, figsize=figsize, sharex=False)

        axes[0, 0].title.set_text("Weight Histograms")
        axes[0, 1].title.set_text("Weight Gradient Histograms")
        axes[0, 2].title.set_text("Bias Histograms")
        axes[0, 3].title.set_text("Bias Gradient Histograms")

        for i in range(4):
            axes[len(self.layer_list) - 1, i].set_xlabel('Value')

        for i, layer in enumerate(self.layer_list):
            axes[i, 0].set_ylabel(self.layer_list_names[i])

            plt.sca(axes[i, 0])
            uu.convert_np_hist_to_plot(self.histogram_weight_history[layer]['weight'][epoch])

            plt.sca(axes[i, 2])
            uu.convert_np_hist_to_plot(self.histogram_weight_history[layer]['bias'][epoch])
            if epoch == 0:
                pass
            else:
                plt.sca(axes[i, 1])
                uu.convert_np_hist_to_plot(self.histogram_gradient_history[layer]['weight'][epoch])

                plt.sca(axes[i, 3])
                uu.convert_np_hist_to_plot(self.histogram_gradient_history[layer]['bias'][epoch])

        sup = self.name + " Layer Weight and Gradient Histograms - Epoch " + str(epoch)
        st = f.suptitle(sup, fontsize='x-large')
        f.tight_layout()
        st.set_y(0.96)
        f.subplots_adjust(top=0.9)

        if show:
            f.show()

        if save:
            assert os.path.exists(save), "Check that the desired save path exists."
            os.makedirs(os.path.join(save, 'layer_histograms'), exist_ok=True)
            f.savefig(os.path.join(save, 'layer_histograms', self.name + '_epoch_' + str(epoch) + '_layer_histograms.png'))

    def build_hist_gif(self, path=None, start=0, stop=None, freq=1, fps=5, final_img_frames=20):
        """
        Loop through self.histogram_weight_history and saves the images to a folder.
        :param path: Path to folder to save images. Folder will be created if it does not already exist.
        :param start: Epoch to start gif on. Default 0.
        :param stop: Epoch to end gif on. Default self.epoch (number of epochs trained so far).
        :param freq: Interval of skipping epochs. Defaults to 1 (no skipping).
        :param fps: Number of frames to display per second in gif. Defaults to 5.
        :param final_img_frames: Number of times to repeat final image of gif before it will restart. Defaults to 20 (4 seconds with 5 fps).
        :return: Saves a gif with the title net + _histogram_generation_animation.gif (as well as the images comprising the gif into the layer_histograms folder)
        """
        assert len(self.histogram_weight_history[self.layer_list[0]]['weight']) > 1, "Model not yet trained"

        if path is None:
            path = self.path

        if stop is None:
            stop = self.epoch

        ims = []
        for epoch in range(start, stop + freq, freq):
            self.plot_layer_hists(epoch=epoch, show=False, save=path)
            img_name = os.path.join(path, 'layer_histograms', self.name + '_epoch_' + str(epoch) + '_layer_histograms.png')
            ims.append(imageio.imread(img_name))
            plt.close()
            if epoch == (stop + freq):  # Hacky method to stay on the final frame for longer
                for i in range(final_img_frames):
                    ims.append(imageio.imread(img_name))
                    plt.close()
        imageio.mimsave(os.path.join(path, self.name + '_histogram_generation_animation.gif'), ims, fps=fps)

    @torch.utils.hooks.unserializable_hook
    def activations_hook(self, grad):
        """
        Used for Grad CAM
        Hook for the gradients of the activations
        Used on the final convolutional layer
        """
        self.gradients = grad

    def get_activations_gradient(self):
        """Grad CAM Helper Function"""
        return self.gradients

    def get_activations(self):
        """Grad CAM Helper Function"""
        return self.final_conv_output


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer"""
    def __init__(self, device, sigma=0.1):
        super().__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma != 0.0:
            sampled_noise = torch.randn(*x.size(), device=self.device) * self.sigma
            x = x + sampled_noise
        return x


class CustomCatGANLayer(nn.Module):
    def __init__(self, cat_mask, le_dict):
        super().__init__()
        # Softmax activation
        self.sm = nn.Softmax(dim=-2)

        # Masks
        self.cat = torch.Tensor(cat_mask).nonzero()
        self.cont = torch.Tensor(~cat_mask).nonzero()

        # Label encoding dictionary
        self.le_dict = le_dict

    def forward(self, input_layer):
        """
        Softmax for each categorical variable - https://medium.com/jungle-book/towards-data-set-augmentation-with-gans-9dd64e9628e6
        :param input_layer: fully connected input layer with size out_dim
        :return: output of forward pass
        """
        cont = input_layer[:, self.cont]

        cat = input_layer[:, self.cat]
        catted = torch.empty_like(cat)
        curr = 0
        for _, le in self.le_dict.items():
            newcurr = curr + len(le.classes_)
            catted[:, curr:newcurr] = self.sm(cat[:, curr:newcurr])
            curr = newcurr

        return torch.cat([catted, cont], 1)

