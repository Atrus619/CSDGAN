import utils.ImageUtils as IU
from CSDGAN.classes.NetUtils import NetUtils, GaussianNoise

import torch.optim as optim
import cv2
import matplotlib
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Discriminator class
class ImageNetD(nn.Module, NetUtils):
    def __init__(self, nf, nc, num_channels, device, path, x_dim, noise=0.0, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super().__init__()
        NetUtils.__init__(self)
        self.name = "Discriminator"

        self.path = path
        self.device = device

        self.loss_real = None
        self.loss_fake = None

        self.nc = nc
        self.nf = nf
        self.num_channels = num_channels
        self.x_dim = x_dim
        self.epoch = 0
        self.fc_labels_size = 128
        self.agg_size = 512

        self.noise = GaussianNoise(device=self.device, sigma=noise)

        # Convolutional layers
        self.arch = OrderedDict()
        self.final_conv_output = None

        # FC layers - Initialized in assemble_architecture method
        self.flattened_dim = None
        self.fc_labels = None
        self.fc_agg = None
        self.fc_output = None

        self.assemble_architecture(h=self.x_dim[0], w=self.x_dim[1])

        # Activations
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Sigmoid()

        # Loss and Optimizer
        self.loss_fn = nn.BCELoss()  # BCE Loss combined with sigmoid for numeric stability
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Initialize weights
        self.weights_init()

        # Record history of training
        self.init_layer_list()
        self.init_history()
        self.update_hist_list()

        self.D_x = []  # Per step
        self.Avg_D_reals = []  # D_x across epochs
        self.D_G_z1 = []  # Per step
        self.Avg_D_fakes = []  # Store D_G_z1 across epochs

        # Grad CAM
        self.gradients = None
        self.final_conv_output = None

    def forward(self, img, labels):
        """
        Deep Convolutional Downsampling Network of Variable Image Size (on creation only)
        layer[0] = Conv2d
        layer[1] = BatchNorm2d
        :param img: Input image of cropped size
        :param labels: Label embedding
        :return: Binary classification (sigmoid activation on a single unit hidden layer)
        """
        x = self.noise(img)

        for i, (layer_name, layer) in enumerate(self.arch.items()):
            if i < (len(self.arch) - 1):
                x = self.act(layer[1](layer[0](x)))
            else:  # Handle final conv layer specially for grad CAM purposes
                self.final_conv_output = layer[0](x)
                self.final_conv_output.requires_grad_()
                h = self.final_conv_output.register_hook(self.activations_hook)
                # Continue
                x = self.act(layer[1](self.final_conv_output))

        x = x.view(-1, self.flattened_dim)
        y = self.act(self.fc_labels(labels))

        agg = torch.cat((x, y), dim=1)
        agg = self.act(self.fc_agg(agg))
        return self.m(self.fc_output(agg))

    def train_one_step_real(self, output, label):
        self.zero_grad()
        self.loss_real = self.loss_fn(output, label)
        self.loss_real.backward()
        self.D_x.append(output.mean().item())

    def train_one_step_fake(self, output, label):
        self.loss_fake = self.loss_fn(output, label)
        self.loss_fake.backward()
        self.D_G_z1.append(output.mean().item())

    def combine_and_update_opt(self):
        self.loss.append(self.loss_real.item() + self.loss_fake.item())
        self.opt.step()
        self.store_weight_and_grad_norms()

    def next_epoch_discrim(self):
        """Discriminator specific actions"""
        self.Avg_D_reals.append(np.mean(self.D_x))  # Mean of means is not exact, but close enough for our purposes
        self.D_x = []

        self.Avg_D_fakes.append(np.mean(self.D_G_z1))
        self.D_G_z1 = []

    def draw_cam(self, img, label, path, real, scale, show=True):
        """
        Implements Grad CAM for netD
        :param img: Image to draw over
        :param label: Corresponding label for img
        :param path: Path to save output image to. Full image path that should end in .jpg
        :param real: Whether the image is real or not
        :param scale: Multiplier to scale image back to original values
        :param show: Whether to show the image
        :return: Pair of images, side by side, left image is drawn over, right image is original
        """
        self.eval()

        # Preprocess inputs
        label = IU.convert_y_to_one_hot(y=torch.full((1, 1), label, dtype=torch.int64), nc=self.nc)
        img, label = img.to(self.device), label.to(self.device)
        img = img.view(-1, 1, self.x_dim[0], self.x_dim[1])
        label = label.view(-1, self.nc).type(torch.float32)

        pred = self.forward(img, label)
        pred.backward()
        gradients = self.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = self.get_activations().detach()

        # Weight the channels by corresponding gradients
        for i in range(self.nf * 2):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu()

        # ReLU on top of the heatmap (possibly should use the actual activation in the network???)
        heatmap = np.maximum(heatmap, 0)

        # Normalize heatmap
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.numpy()

        # Save original image
        img_transformed = img.view(self.x_dim[0], self.x_dim[1]).detach().cpu().numpy() * scale
        matplotlib.image.imsave(path, img_transformed, cmap='gray')

        # Read in image and cut pixels in half for visibility
        cv_img = cv2.imread(path)
        cv_img = cv_img / 2

        # Create heatmap
        heatmap = cv2.resize(heatmap, (cv_img.shape[1], cv_img.shape[0]))
        heatmap = np.uint8(scale * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose
        superimposed_img = heatmap * 0.4 + cv_img

        # Save
        cv2.imwrite(path, superimposed_img)

        # Load in and make pretty
        show_img = plt.imread(path)
        f, axes = plt.subplots(1, 2, figsize=(14, 7))
        plt.sca(axes[0])
        plt.axis('off')
        plt.title('Grad CAM', fontweight='bold')
        plt.imshow(show_img)

        plt.sca(axes[1])
        plt.axis('off')
        plt.title('Original Image', fontweight='bold')
        img = img.squeeze().detach().cpu().numpy()
        plt.imshow(img, cmap='gray')

        real_or_fake = 'real' if pred > 0.5 else 'fake'
        real_str = 'Real' if real else 'Fake'
        sup = 'Discriminator Gradient Class Activation Map\n\n' + real_str + ' image predicted to be ' + real_or_fake
        st = f.suptitle(sup, fontsize='x-large', fontweight='bold')
        f.tight_layout()
        st.set_y(0.96)
        f.subplots_adjust(top=0.8)

        f.savefig(path)

        if show:
            plt.show()

    def assemble_architecture(self, h, w):
        """Fills in an ordered dictionaries with tuples, one for the layers and one for the corresponding batch norm layers"""
        h_best_crop, h_best_first, h_pow_2 = IU.find_pow_2_arch(h)
        w_best_crop, w_best_first, w_pow_2 = IU.find_pow_2_arch(w)
        assert (h_best_crop, w_best_crop) == (0, 0), "Crop not working properly"

        # Conv Layers
        num_intermediate_downsample_layers = max(h_pow_2, w_pow_2) - 1

        h_rem, w_rem = self.x_dim[0] - h_best_crop, self.x_dim[1] - w_best_crop
        h_rem, w_rem = h_rem // h_best_first, w_rem // w_best_first

        h_rem, w_rem, h_curr, w_curr = IU.update_h_w_curr(h_rem=h_rem, w_rem=w_rem)
        self.arch['cn1'] = IU.cn2_downsample_block(h=h_curr, w=w_curr, in_channels=self.num_channels, out_channels=self.nf)
        self.add_module('cn1', self.arch['cn1'][0])
        self.add_module('cn1_bn', self.arch['cn1'][1])

        # Downsample by 2x until it is no longer necessary, then downsample by 1x
        for i in range(num_intermediate_downsample_layers):
            h_rem, w_rem, h_curr, w_curr = IU.update_h_w_curr(h_rem=h_rem, w_rem=w_rem)
            self.arch['cn' + str(i + 2)] = IU.cn2_downsample_block(h=h_curr, w=w_curr,
                                                                   in_channels=self.nf * 2 ** i,
                                                                   out_channels=self.nf * 2 ** (i + 1))
            self.add_module('cn' + str(i + 2), self.arch['cn' + str(i + 2)][0])
            self.add_module('cn' + str(i + 2) + '_bn', self.arch['cn' + str(i + 2)][1])

        # FC Layers
        self.fc_labels = nn.Linear(in_features=self.nc, out_features=self.fc_labels_size, bias=True)
        self.flattened_dim = self.nf * 2 ** num_intermediate_downsample_layers * h_best_first * w_best_first
        self.fc_agg = nn.Linear(in_features=self.flattened_dim + self.fc_labels_size,
                                out_features=self.agg_size, bias=True)
        self.fc_output = nn.Linear(in_features=self.agg_size, out_features=1, bias=True)
