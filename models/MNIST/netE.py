import torch.nn as nn
import torch
from models.NetUtils import NetUtils
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from utils.utils import safe_mkdir
import cv2
import matplotlib


# Evaluator class
class CGAN_Evaluator(nn.Module, NetUtils):
    def __init__(self, train_gen, val_gen, test_gen, device, x_dim, num_channels, nc, lr, beta1, beta2, wd):
        super().__init__()

        self.nc = nc
        self.nf = 10  # Completely arbitrary. Works well for now.
        self.x_dim = x_dim

        # Generators
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen

        self.device = device

        # Layers
        self.cn1 = nn.Conv2d(in_channels=num_channels, out_channels=self.nf, kernel_size=5, stride=1, padding=2, bias=True)
        self.cn1_bn = nn.BatchNorm2d(10)
        self.cn2 = nn.Conv2d(in_channels=self.nf, out_channels=self.nf*2, kernel_size=5, stride=1, padding=2, bias=True)
        self.cn2_bn = nn.BatchNorm2d(20)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=14 * 14 * 20, out_features=64)
        self.output = nn.Linear(in_features=64, out_features=self.nc)

        # Activations
        self.do2d = nn.Dropout2d(0.2)
        self.do1d = nn.Dropout(0.2)
        self.act = nn.LeakyReLU(0.2)

        # Loss and Optimizer
        self.loss = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Initialize weights
        self.weights_init()

        # Record history
        self.train_losses = []
        self.val_losses = []
        self.val_acc = []

        # Grad CAM
        self.gradients = None
        self.final_conv_output = None

    def forward(self, x):
        x = self.do2d(self.act(self.cn1_bn(self.cn1(x))))

        # Register hook for Grad CAM
        self.final_conv_output = self.cn2(x)
        self.final_conv_output.requires_grad_()
        h = self.final_conv_output.register_hook(self.activations_hook)

        # Continue
        x = self.do2d(self.act(self.cn2_bn(self.final_conv_output)))
        x = self.mp(x)
        x = x.view(-1, 14 * 14 * 20)
        x = self.do1d(self.act(self.fc1(x)))
        return self.output(x)  # No softmax activation needed because it is built into CrossEntropyLoss in pytorch

    def train_one_epoch(self):
        self.train()
        train_loss = 0
        running_count = 0
        for batch, labels in self.train_gen:
            # Forward pass
            batch, labels = batch.to(self.device), labels.to(self.device)
            self.zero_grad()
            fwd = self.forward(batch)
            # Calculate loss and update optimizer
            label_ind = torch.argmax(labels, -1)
            self.loss = self.loss_fn(fwd, label_ind)
            self.loss.backward()
            self.opt.step()
            # Update running totals
            train_loss += self.loss
            running_count += len(batch)
        return train_loss / running_count

    def eval_once(self, gen):
        self.eval()
        val_loss = 0
        num_correct = torch.zeros(1)
        running_count = 0
        with torch.no_grad():
            for batch, labels in gen:
                # Forward pass
                batch, labels = batch.to(self.device), labels.to(self.device)
                fwd = self.forward(batch)
                # Calculate loss and accuracy, and update running totals
                label_ind = torch.argmax(labels, -1)
                val_loss += self.loss_fn(fwd, label_ind)
                num_correct += sum(torch.argmax(fwd, -1) == label_ind)
                running_count += len(batch)
        return val_loss / running_count, num_correct / running_count

    def train_evaluator(self, num_epochs, eval_freq, es=None):
        for epoch in range(num_epochs):
            total_loss = self.train_one_epoch()
            self.train_losses.append(total_loss.item())

            if epoch % eval_freq == 0 or (epoch == num_epochs - 1):
                total_loss, total_acc = self.eval_once(self.val_gen)
                self.val_losses.append(total_loss.item())
                self.val_acc.append(total_acc.item())

                if es:
                    if np.argmin(self.val_losses) < (epoch - es + 1):
                        return True  # Exit early
        return True

    def classification_stats(self, title='', show=True, save=None):
        """Return a confusion matrix, classification report, and plots/saves a heatmap confusion matrix on the predictions"""
        self.eval()

        ground_truth = torch.empty(size=(0, 0), dtype=torch.int64, device=self.device).view(-1)
        preds = torch.empty_like(ground_truth)

        with torch.no_grad():
            for batch, labels in self.test_gen:
                batch, labels = batch.to(self.device), labels.to(self.device)
                fwd = self.forward(batch)

                ground_truth = torch.cat((ground_truth, torch.argmax(labels, -1)), dim=0)
                preds = torch.cat((preds, torch.argmax(fwd, -1)), dim=0)

        cm = confusion_matrix(ground_truth.detach().cpu().numpy(), preds.detach().cpu().numpy())
        cr = classification_report(ground_truth.detach().cpu().numpy(), preds.detach().cpu().numpy())

        plt.figure(figsize=(10, 7))
        df_cm = pd.DataFrame(cm, index=list(range(self.nc)), columns=list(range(self.nc)))
        sns.heatmap(df_cm, annot=True)

        if show:
            plt.show()

        if save is not None:
            assert os.path.exists(save), "Check that the desired save path exists."
            safe_mkdir(save + '/conf_heatmaps')
            plt.savefig(save + '/conf_heatmaps/' + title + '_conf_heatmap.png')

        return cm, cr

    def draw_cam(self, img, path, show=True):
        """
        Implements Grad CAM for netE
        :param img: Image to draw over
        :param path: Path to save output image to
        :param show: Whether to show the image
        :return: Pair of images, side by side, left image is drawn over, right image is original
        """
        self.eval()

        # Preprocess inputs
        img = img.to(self.device)
        img = img.view(-1, 1, self.x_dim[0], self.x_dim[1])

        pred = self.forward(img)
        pred[:, pred.argmax(1)].backward()
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

            sup = 'Evaluator Gradient Class Activation Map\n\nPredicted to be ' + str(pred.argmax(1).detach().cpu().numpy().take(0))
            st = f.suptitle(sup, fontsize='x-large', fontweight='bold')
            f.tight_layout()
            st.set_y(0.96)
            f.subplots_adjust(top=0.8)

            plt.show()
