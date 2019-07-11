import torch.nn as nn
import torch
from models.NetUtils import NetUtils, GaussianNoise
import torch.optim as optim
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from utils.MNIST import convert_y_to_one_hot


# Discriminator class
class CGAN_Discriminator(nn.Module, NetUtils):
    def __init__(self, nf, nc, num_channels, device, x_dim, noise=0.0, lr=2e-4, beta1=0.5, beta2=0.999, wd=0):
        super().__init__()
        NetUtils.__init__(self)

        self.loss_real = None
        self.loss_fake = None
        self.device = device

        self.nc = nc
        self.nf = nf
        self.x_dim = x_dim
        self.epoch = 0
        self.fc_labels_size = 128
        self.agg_size = 512

        self.noise = GaussianNoise(device=self.device, sigma=noise)

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
        :param img: Input image of size 28 x 28
        :param labels: Label embedding
        :return: Binary classification (sigmoid activation on a single unit hidden layer)
        """
        img = self.noise(img)
        x = self.act(self.cn1_bn(self.cn1(img)))

        # Register hook for Grad CAM
        self.final_conv_output = self.cn2(x)
        self.final_conv_output.requires_grad_()
        h = self.final_conv_output.register_hook(self.activations_hook)

        # Continue
        x = self.act(self.cn2_bn(self.final_conv_output))
        x = x.view(-1, self.nf * 2 * 7 * 7)
        y = self.act(self.fc_labels(labels))
        agg = torch.cat((x, y), 1)
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

    def draw_cam(self, img, label, path, show=True):
        """
        Implements Grad CAM for netD
        :param img: Image to draw over
        :param label: Corresponding label for img
        :param path: Path to save output image to
        :param show: Whether to show the image
        :return: Pair of images, side by side, left image is drawn over, right image is original
        """
        self.eval()

        # Preprocess inputs
        label = convert_y_to_one_hot(torch.full((1, 1), label, dtype=torch.int64))
        img, label = img.to(self.device), label.to(self.device)
        img = img.view(-1, 1, self.x_dim[0], self.x_dim[1])
        label = label.view(-1, self.nc).type(torch.float32)

        pred = self.forward(img, label)
        pred.backward()
        gradients = self.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = self.get_activations().detach()

        # Weight the channels by corresponding gradients
        for i in range(self.nf*2):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu()

        # ReLU on top of the heatmap (possibly should use the actual activation in the network???)
        heatmap = np.maximum(heatmap, 0)

        # Normalize heatmap
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.numpy()

        # Save original image
        img_transformed = img.view(self.x_dim[0], self.x_dim[1]).detach().cpu().numpy()*255
        matplotlib.image.imsave(path, img_transformed, cmap='gray')

        # Read in image and cut pixels in half for visibility
        cv_img = cv2.imread(path)
        cv_img = cv_img / 2

        # Create heatmap
        heatmap = cv2.resize(heatmap, (cv_img.shape[1], cv_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose
        superimposed_img = heatmap * 0.4 + cv_img

        # Save
        cv2.imwrite(path, superimposed_img)

        # Load in to show
        if show:
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
            sup = 'Discriminator Gradient Class Activation Map\n\nPredicted to be ' + real_or_fake
            st = f.suptitle(sup, fontsize='x-large', fontweight='bold')
            f.tight_layout()
            st.set_y(0.96)
            f.subplots_adjust(top=0.8)

            plt.show()
