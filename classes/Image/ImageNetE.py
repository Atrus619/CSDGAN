from classes.NetUtils import NetUtils
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import os
from utils.utils import safe_mkdir
import cv2
import matplotlib
from collections import OrderedDict
from utils.ImageUtils import *


# Evaluator class
class ImageNetE(nn.Module, NetUtils):
    def __init__(self, train_gen, val_gen, test_gen, device, path, x_dim, num_channels, nc, le, lr, beta1, beta2, wd):
        super().__init__()
        self.name = "Evaluator"

        self.path = path
        self.nc = nc
        self.nf = 10  # Completely arbitrary. Works well for now.
        self.fc_features = 64  # Completely arbitrary also.
        self.num_channels = num_channels
        self.x_dim = x_dim
        self.le = le

        # Generators
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen

        self.device = device

        # Layers
        self.arch = OrderedDict()
        self.final_conv_output = None

        # FC layers - Initialized in assemble_architecture method
        self.flattened_dim = None
        self.fc1 = None
        self.output = None

        self.assemble_architecture(h=self.x_dim[0], w=self.x_dim[1])

        # Activations/Dropouts
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
        """
        Deep Convolutional Network of Variable Image Size (on creation only)
        Using Max Pooling for Downsampling
        layer[0] = Conv2d
        layer[1] = BatchNorm2d
        layer[2] = MaxPool2d
        """
        for i, (layer_name, layer) in enumerate(self.arch.items()):
            if i < (len(self.arch) - 1):
                x = layer[2](self.do2d(self.act(layer[1](layer[0](x)))))
            else:  # Handle final conv layer specially for grad CAM purposes
                self.final_conv_output = layer[0](x)
                self.final_conv_output.requires_grad_()
                h = self.final_conv_output.register_hook(self.activations_hook)
                # Continue
                x = layer[2](self.do2d(self.act(layer[1](self.final_conv_output))))

        x = x.view(-1, self.flattened_dim)
        x = self.do1d(self.act(self.fc1(x)))
        return self.output(x)  # No softmax activation needed because it is built into CrossEntropyLoss in pytorch

    def train_one_epoch_real(self):
        self.train()
        train_loss = 0
        running_count = 0
        for batch, labels in self.train_gen:
            self.train_step(batch=batch, labels=labels)
            # Update running totals
            train_loss += self.loss
            running_count += len(batch)
        return train_loss / running_count

    def train_one_epoch_fake(self):
        self.train()
        train_loss = 0
        running_count = 0
        self.train_gen.dataset.next_epoch()
        for i in range(self.train_gen.dataset.batches_per_epoch):
            batch, labels = self.train_gen.dataset.next_batch()
            self.train_step(batch=batch, labels=labels)
            # Update running totals
            train_loss += self.loss
            running_count += len(batch)
        return train_loss / running_count

    def train_step(self, batch, labels):
        # Forward pass
        batch, labels = batch.to(self.device), labels.to(self.device)
        self.zero_grad()
        fwd = self.forward(batch)
        # Calculate loss and update optimizer
        label_ind = torch.argmax(labels, -1)
        self.loss = self.loss_fn(fwd, label_ind)
        self.loss.backward()
        self.opt.step()

    def eval_once_real(self, gen):
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

    def eval_once_fake(self, gen):
        self.eval()
        val_loss = 0
        num_correct = torch.zeros(1)
        running_count = 0
        gen.dataset.next_epoch()
        with torch.no_grad():
            for i in range(gen.dataset.batches_per_epoch):
                batch, labels = gen.dataset.next_batch()
                # Forward pass
                batch, labels = batch.to(self.device), labels.to(self.device)
                fwd = self.forward(batch)
                # Calculate loss and accuracy, and update running totals
                label_ind = torch.argmax(labels, -1)
                val_loss += self.loss_fn(fwd, label_ind)
                num_correct += sum(torch.argmax(fwd, -1) == label_ind)
                running_count += len(batch)
        return val_loss / running_count, num_correct / running_count

    def train_evaluator(self, num_epochs, eval_freq, real, es=None):
        for epoch in range(num_epochs):
            if real:
                total_loss = self.train_one_epoch_real()
            else:
                total_loss = self.train_one_epoch_fake()
            self.train_losses.append(total_loss.item())

            if epoch % eval_freq == 0 or (epoch == num_epochs - 1):
                if real:
                    total_loss, total_acc = self.eval_once_real(gen=self.val_gen)
                else:
                    total_loss, total_acc = self.eval_once_fake(gen=self.val_gen)
                self.val_losses.append(total_loss.item())
                self.val_acc.append(total_acc.item())

                if es:
                    if np.argmin(self.val_losses) < (epoch - es + 1):
                        return True  # Exit early
        return True

    def classification_stats(self, title='', show=True, save=None):
        """Return a confusion matrix, classification report, and plots/saves a heatmap confusion matrix on the predictions"""
        if save is None:
            save = self.path

        ground_truth = torch.empty(size=(0, 0), dtype=torch.int64, device=self.device).view(-1)
        preds = torch.empty_like(ground_truth)

        self.eval()
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

        if save:
            assert os.path.exists(save), "Check that the desired save path exists."
            safe_mkdir(save + '/conf_heatmaps')
            plt.savefig(save + '/conf_heatmaps/' + title + '_conf_heatmap.png')

        return cm, cr

    def draw_cam(self, img, path, real, scale, show=True):
        """
        Implements Grad CAM for netE
        :param img: Image to draw over
        :param path: Path to save output image to. Full image path that should end in .jpg
        :param real: Whether the image is real or not
        :param scale: Multiplier to scale image back to original values
        :param show: Whether to show the image
        :return: Pair of images, side by side, left image is drawn over, right image is original
        """
        # Preprocess inputs
        img = img.to(self.device)
        img = img.view(-1, 1, self.x_dim[0], self.x_dim[1])

        self.eval()

        pred = self.forward(img)
        pred[:, pred.argmax(1)].backward()
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

        # Load in to show

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

        real_str = 'Real' if real else 'Fake'
        pred_index = pred.argmax(1).detach().cpu().numpy().take(0)
        sup = 'Evaluator Gradient Class Activation Map\n\n' + real_str + ' image predicted to be ' + str(self.le.inverse_transform([pred_index]).take(0))
        st = f.suptitle(sup, fontsize='x-large', fontweight='bold')
        f.tight_layout()
        st.set_y(0.96)
        f.subplots_adjust(top=0.8)

        f.savefig(path)

        if show:
            plt.show()

    def assemble_architecture(self, h, w):
        """Fills in an ordered dictionaries with tuples, one for the layers and one for the corresponding batch norm layers"""
        h_best_crop, h_best_first, h_pow_2 = find_pow_2_arch(h)
        w_best_crop, w_best_first, w_pow_2 = find_pow_2_arch(w)
        assert (h_best_crop, w_best_crop) == (0, 0), "Crop not working properly"

        # Conv Layers
        num_intermediate_downsample_layers = max(h_pow_2, w_pow_2) - 1

        h_rem, w_rem = self.x_dim[0] - h_best_crop, self.x_dim[1] - w_best_crop
        h_rem, w_rem = h_rem // h_best_first, w_rem // w_best_first

        h_rem, w_rem, h_curr, w_curr = update_h_w_curr(h_rem=h_rem, w_rem=w_rem)
        self.arch['cn1'] = evaluator_cn2_block(h=h_curr, w=w_curr, in_channels=self.num_channels, out_channels=self.nf)
        self.add_module('cn1', self.arch['cn1'][0])
        self.add_module('cn1_bn', self.arch['cn1'][1])
        self.add_module('cn1_mp', self.arch['cn1'][2])

        # For the evaluator, we will use max pooling, so we will build layers that perform no downsampling
        for i in range(num_intermediate_downsample_layers):
            h_rem, w_rem, h_curr, w_curr = update_h_w_curr(h_rem=h_rem, w_rem=w_rem)
            self.arch['cn' + str(i + 2)] = evaluator_cn2_block(h=h_curr, w=w_curr,
                                                               in_channels=self.nf * 2 ** i,
                                                               out_channels=self.nf * 2 ** (i + 1))
            self.add_module('cn' + str(i + 2), self.arch['cn' + str(i + 2)][0])
            self.add_module('cn' + str(i + 2) + '_bn', self.arch['cn' + str(i + 2)][1])
            self.add_module('cn' + str(i + 2) + '_mp', self.arch['cn' + str(i + 2)][2])

        # FC Layers
        self.flattened_dim = self.nf * 2 ** num_intermediate_downsample_layers * h_best_first * w_best_first
        self.fc1 = nn.Linear(in_features=self.flattened_dim, out_features=self.fc_features)
        self.output = nn.Linear(in_features=self.fc_features, out_features=self.nc)
