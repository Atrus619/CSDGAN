import CSDGAN.utils.db as db
import CSDGAN.utils.constants as cs
import utils.image_utils as iu
import utils.utils as uu
from CSDGAN.classes.image.ImageDataset import OnlineGeneratedImageDataset
from CSDGAN.classes.image.ImageNetD import ImageNetD
from CSDGAN.classes.image.ImageNetG import ImageNetG
from CSDGAN.classes.image.ImageNetE import ImageNetE
from CSDGAN.classes.NetUtils import GaussianNoise
from CSDGAN.classes.CGANUtils import CGANUtils

import time
from torch.utils import data
import imageio
import copy
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils


class ImageCGAN(CGANUtils):
    """CGAN for image-based data sets"""

    def __init__(self, train_gen, val_gen, test_gen, device, nc, nz, num_channels, sched_netG, path, le, ohe,
                 label_noise, label_noise_linear_anneal, discrim_noise, discrim_noise_linear_anneal,
                 netG_nf, netG_lr, netG_beta1, netG_beta2, netG_wd,
                 netD_nf, netD_lr, netD_beta1, netD_beta2, netD_wd,
                 netE_lr, netE_beta1, netE_beta2, netE_wd,
                 fake_data_set_size, fake_bs,
                 eval_num_epochs, early_stopping_patience, grid_num_examples=10):
        super().__init__()

        self.path = path  # default file path for saved objects
        self.init_paths()

        # Data generator
        self.train_gen = train_gen
        self.data_gen = self.train_gen  # For drawing architectures only
        self.val_gen = val_gen
        self.test_gen = test_gen

        # Initialize properties
        self.device = device
        self.x_dim = self.extract_x_dim()
        self.nc = nc
        self.nz = nz
        self.num_channels = num_channels

        self.le = le
        self.ohe = ohe
        self.grid_num_examples = grid_num_examples

        # Anti-discriminator properties
        assert 0.0 <= label_noise <= 1.0, "Label noise must be between 0 and 1"
        self.label_noise = label_noise
        self.label_noise_linear_anneal = label_noise_linear_anneal
        self.ln_rate = 0.0

        self.discrim_noise = discrim_noise
        self.discrim_noise_linear_anneal = discrim_noise_linear_anneal
        self.dn_rate = 0.0

        # Evaluator properties
        self.fake_shuffle = True
        self.fake_num_workers = 6
        self.fake_data_set_size = fake_data_set_size
        self.fake_bs = fake_bs

        self.netE_params = {'lr': netE_lr, 'beta1': netE_beta1, 'beta2': netE_beta2, 'wd': netE_wd}

        self.eval_num_epochs = eval_num_epochs
        self.early_stopping_patience = early_stopping_patience

        # Initialized through init_fake_gen method
        self.fake_train_set = None
        self.fake_train_gen = None
        self.fake_val_set = None
        self.fake_val_gen = None

        # Instantiate sub-nets
        self.netG = ImageNetG(nz=self.nz, num_channels=self.num_channels, nf=netG_nf, x_dim=self.x_dim, nc=self.nc, device=self.device, path=self.path,
                              grid_num_examples=self.grid_num_examples, lr=netG_lr, beta1=netG_beta1, beta2=netG_beta2, wd=netG_wd).to(self.device)
        self.netD = ImageNetD(nf=netD_nf, num_channels=self.num_channels, nc=self.nc, noise=self.discrim_noise, device=self.device, x_dim=self.x_dim,
                              path=self.path, lr=netD_lr, beta1=netD_beta1, beta2=netD_beta2, wd=netD_wd).to(self.device)
        self.netE = None  # Initialized through init_evaluator method
        self.nets = {self.netG, self.netD, self.netE}

        # Training properties
        self.epoch = 0
        self.sched_netG = sched_netG
        self.real_label = 1
        self.fake_label = 0
        self.stored_loss = []
        self.stored_acc = []

        self.fixed_imgs = [self.gen_fixed_img_grid()]

    def train_gan(self, num_epochs, print_freq, eval_freq=None, run_id=None, logger=None):
        """
        Primary method for training
        :param num_epochs: Desired number of epochs to train for
        :param print_freq: How frequently to print out training statistics (i.e., freq of 5 will result in information being printed every 5 epochs)
        :param eval_freq: How frequently to evaluate with netE. If None, no evaluation will occur. Evaluation takes a significant amount of time.
        :param run_id: If not None, will update database as it progresses through training in quarter increments.
        :param logger: Logger to be used for logging training progress. Must exist if run_id is not None.
        """
        assert logger if run_id else True, "Must pass a logger if run_id is passed"

        total_epochs = self.epoch + num_epochs

        if run_id:
            checkpoints = [int(num_epochs * i / 4) for i in range(1, 4)]

        if self.label_noise_linear_anneal:
            self.ln_rate = self.label_noise / num_epochs

        if self.discrim_noise_linear_anneal:
            self.dn_rate = self.discrim_noise / num_epochs

        uu.train_log_print(run_id=run_id, logger=logger, statement="Beginning training")
        og_start_time = time.time()
        start_time = time.time()

        for epoch in range(num_epochs):
            for x, y in self.train_gen:
                y = torch.eye(self.nc)[y] if len(y.shape) == 1 else y
                x, y = x.to(self.device), y.to(self.device)
                self.train_one_step(x, y)

            self.next_epoch()

            if self.epoch % print_freq == 0 or (self.epoch == num_epochs):
                uu.train_log_print(run_id=run_id, logger=logger, statement="Time: %ds" % (time.time() - start_time))
                start_time = time.time()

                self.print_progress(total_epochs=total_epochs, run_id=run_id, logger=logger)

            if eval_freq is not None:
                if self.epoch % eval_freq == 0 or (self.epoch == num_epochs):
                    self.init_fake_gen()
                    self.test_model(train_gen=self.fake_train_gen, val_gen=self.fake_val_gen)
                    uu.train_log_print(run_id=run_id, logger=logger, statement="Epoch: %d\tEvaluator Score: %.4f" % (self.epoch, np.max(self.stored_acc[-1])))

            if run_id:
                if self.epoch in checkpoints:
                    db.query_verify_live_run(run_id=run_id)
                    logger.info('Checkpoint reached.')
                    status_id = 'Train ' + str(checkpoints.index(self.epoch) + 1) + '/4'
                    db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT[status_id])

        uu.train_log_print(run_id=run_id, logger=logger, statement="Total training time: %ds" % (time.time() - og_start_time))
        uu.train_log_print(run_id=run_id, logger=logger, statement="Training complete")

    def test_model(self, train_gen, val_gen):
        """
        Train a CNN evaluator from scratch
        :param train_gen: Specified train_gen, can either be real training generator or a created one from netG
        :param val_gen: Same as above ^
        """
        self.init_evaluator(train_gen, val_gen)
        self.netE.train_evaluator(num_epochs=self.eval_num_epochs, eval_freq=1, real=False, es=self.early_stopping_patience)
        torch.save(self.netG.state_dict(), self.path + "/stored_generators/Epoch_" + str(self.epoch) + "_Generator.pt")
        loss, acc = self.netE.eval_once_real(self.test_gen)
        self.stored_loss.append(loss.item())
        self.stored_acc.append(acc.item())

    def next_epoch(self):
        """Run netG and netD methods to prepare for next epoch. Mostly saves histories and resets history collection objects."""
        self.epoch += 1

        self.fixed_imgs.append(self.gen_fixed_img_grid())

        self.netG.next_epoch()
        self.netG.next_epoch_gen()

        self.netD.next_epoch()
        self.netD.next_epoch_discrim()

        # Anneal noise rates
        self.label_noise -= self.ln_rate
        self.discrim_noise -= self.dn_rate
        self.netD.noise = GaussianNoise(device=self.device, sigma=self.discrim_noise)

    def init_evaluator(self, train_gen, val_gen):
        """
        Initialize the netE sub-net. This is done as a separate method because we want to reinitialize netE each time we want to evaluate it.
        We can also evaluate on the original, real data by specifying these training generators.
        """
        self.netE = ImageNetE(train_gen=train_gen, val_gen=val_gen, test_gen=self.test_gen, device=self.device, x_dim=self.x_dim, le=self.le,
                              num_channels=self.num_channels, nc=self.nc, path=self.path, **self.netE_params).to(self.device)
        self.nets = {self.netG, self.netD, self.netE}

    def init_fake_gen(self):
        # Initialize fake training set and validation set to be same size
        self.fake_train_set = OnlineGeneratedImageDataset(netG=self.netG, size=self.fake_data_set_size, nz=self.nz, nc=self.nc, bs=self.fake_bs,
                                                          ohe=self.ohe, device=self.device, x_dim=self.x_dim)
        self.fake_train_gen = data.DataLoader(self.fake_train_set, batch_size=self.fake_bs,
                                              shuffle=self.fake_shuffle, num_workers=self.fake_num_workers)

        self.fake_val_set = OnlineGeneratedImageDataset(netG=self.netG, size=self.fake_data_set_size, nz=self.nz, nc=self.nc, bs=self.fake_bs,
                                                        ohe=self.ohe, device=self.device, x_dim=self.x_dim)
        self.fake_val_gen = data.DataLoader(self.fake_val_set, batch_size=self.fake_bs,
                                            shuffle=self.fake_shuffle, num_workers=self.fake_num_workers)

    def eval_on_real_data(self, num_epochs, train_gen=None, val_gen=None, test_gen=None, es=None):
        """
        Evaluate the CGAN Evaluator Network on real examples
        :param num_epochs: Number of epochs to train for
        :param train_gen: PyTorch generator
        :param val_gen: PyTorch generator
        :param test_gen: PyTorch generator
        :param es: Early-stopping patience. If None, early-stopping is not utilized.
        :return: Accuracy of evaluation on CGAN's testing data
        """
        if train_gen is None:
            train_gen = self.train_gen

        if val_gen is None:
            val_gen = self.val_gen

        if test_gen is None:
            test_gen = self.test_gen

        self.init_evaluator(train_gen, val_gen)
        self.netE.train_evaluator(num_epochs=num_epochs, eval_freq=1, real=True, es=es)
        _, og_result = self.netE.eval_once_real(test_gen)
        og_result = og_result.numpy().take(0)
        return og_result, copy.copy(self.netE)

    def show_img(self, label):
        """Generate an image based on the desired class label index (integer 0-9)"""
        assert label in self.le.classes_, "Make sure label is a valid class"
        label = self.le.transform([label])[0]
        label = torch.full((1, 1), label, dtype=torch.int64)

        noise = torch.randn(1, self.nz, device=self.device)
        processed_label = torch.zeros([1, self.nc], dtype=torch.uint8, device='cpu')
        processed_label = processed_label.scatter(1, label, 1).float().to(self.device)

        self.netG.eval()
        with torch.no_grad():
            output = self.netG(noise, processed_label).view(self.num_channels, self.x_dim[0], self.x_dim[1]).detach().cpu()

        plt.imshow(output.permute(1, 2, 0))
        plt.show()

    def gen_fixed_img_grid(self):
        """
        Produce a grid of generated images from netG's fixed noise vector. This can be used to visually track progress of the CGAN training.
        :return: Tensor of images
        """
        self.netG.eval()
        with torch.no_grad():
            fixed_imgs = self.netG(self.netG.fixed_noise, self.netG.fixed_labels)
        return vutils.make_grid(tensor=fixed_imgs, nrow=self.grid_num_examples, normalize=True).detach().cpu()

    def get_grid(self, index=-1, labels=None, num_examples=None):
        """Same as show_grid, but produces the specific grid (helper function)"""
        # Check inputs
        assert len(self.fixed_imgs) > 0, 'Model not yet trained'

        if num_examples is None:
            num_examples = self.grid_num_examples
        assert num_examples <= self.grid_num_examples, 'Num examples must be less than or equal to ' + str(self.grid_num_examples)

        if labels is None:
            labels = self.le.classes_

        # Instantiate output object
        og_img = self.fixed_imgs[index]
        new_img = torch.zeros([og_img.shape[0], len(labels) * self.x_dim[0] + 2 * (1 + len(labels)), num_examples * self.x_dim[1] + 2 * (1 + num_examples)],
                              dtype=torch.float32)

        # Fill in new_img with relevant parts of og_img
        for i, label in enumerate(labels):
            for j in range(num_examples):
                start_loc = np.where(label == self.le.classes_)[0][0]
                new_img[:, i * self.x_dim[0] + 2 * (1 + i):(1 + i) * self.x_dim[0] + 2 * (2 + i), j * self.x_dim[1] + 2 * (1 + j):(1 + j) * self.x_dim[1] + 2 * (2 + j)] = \
                    og_img[:, start_loc * self.x_dim[0] + 2 * (1 + start_loc):(1 + start_loc) * self.x_dim[0] + 2 * (2 + start_loc),
                    j * self.x_dim[1] + 2 * (1 + j):(1 + j) * self.x_dim[1] + 2 * (2 + j)]

        return new_img

    def show_grid(self, index=-1, labels=None, num_examples=None):
        """
        Print a specified fixed image grid from the self.fixed_imgs list
        :param index: Evaluation index to display
        :param labels: Which categories to show grid for
        :param num_examples: Number of examples of each category to include in grid
        :return: Nothing. Displays the desired image instead.
        """
        # Check inputs
        assert len(self.fixed_imgs) > 0, 'Model not yet trained'

        if num_examples is None:
            num_examples = self.grid_num_examples
        assert num_examples <= self.grid_num_examples, 'Num examples must be less than or equal to ' + str(self.grid_num_examples)

        if labels is None:
            labels = self.le.classes_

        # Get img
        new_img = self.get_grid(index=index, labels=labels, num_examples=num_examples)

        # Show img
        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(np.transpose(new_img, (1, 2, 0)))
        plt.show()

    def build_gif(self, labels=None, num_examples=None, path=None, start=0, stop=None, freq=1, fps=5, final_img_frames=20):
        """
        Loop through self.fixed_imgs and saves the images to a folder.
        :param labels: List of which labels to produce. Defaults to all.
        :param num_examples: Number of each label to produce. Defaults to self.grid_num_examples (10 generally).
        :param path: Path to folder to save images. Folder will be created if it does not already exist.
        :param start: Epoch to start gif on. Default 0.
        :param stop: Epoch to end gif on. Default self.epoch (number of epochs trained so far).
        :param freq: Interval of skipping epochs. Defaults to 1 (no skipping).
        :param fps: Number of frames to display per second in gif. Defaults to 5.
        :param final_img_frames: Number of times to repeat final image of gif before it will restart. Defaults to 20 (4 seconds with 5 fps).
        :return: GIF (large file size, be careful!).
        """
        # Check inputs
        assert len(self.fixed_imgs) > 0, 'Model not yet trained'

        if num_examples is None:
            num_examples = self.grid_num_examples
        assert num_examples <= self.grid_num_examples, 'Num examples must be less than or equal to ' + str(self.grid_num_examples)

        if labels is None:
            labels = self.le.classes_

        if path is None:
            path = self.path

        if stop is None:
            stop = self.epoch

        os.makedirs(os.path.join(path, "imgs"), exist_ok=True)
        ims = []
        for epoch in range(start, self.epoch + 1, freq):
            fig = plt.figure(figsize=(8, 8))
            plt.axis('off')
            plt.suptitle('Epoch ' + str(epoch))
            grid = self.get_grid(index=epoch, labels=labels, num_examples=num_examples)
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            img_name = os.path.join(path, 'imgs', 'Epoch ' + str(epoch) + '.png')
            plt.savefig(img_name)
            ims.append(imageio.imread(img_name))
            plt.close()
            if epoch == self.epoch:  # Hacky method to stay on the final frame for longer
                for i in range(final_img_frames):
                    ims.append(imageio.imread(img_name))
                    plt.close()
        imageio.mimsave(os.path.join(path, 'generation_animation.gif'), ims, fps=fps)

    def run_all_diagnostics(self, real_netE, benchmark_acc, show=False, save=None):
        """
        Run all diagnostic methods
        :param real_netE: netE trained on real data
        :param benchmark_acc: Best score obtained from training Evaluator on real data
        :param show: Whether to display the plots as well
        :param save: Where to save the plot. If set to None, default path is used.
        """
        if save is None:
            save = self.path

        self.plot_progress(benchmark_acc=benchmark_acc, show=show, save=save)

        self.build_gif(path=save)
        self.netG.build_hist_gif(path=save)
        self.netD.build_hist_gif(path=save)

        self.plot_training_plots(show=show, save=save)

        self.netG.plot_layer_scatters(show=show, save=save)
        self.netD.plot_layer_scatters(show=show, save=save)

        self.netG.plot_layer_hists(show=show, save=save)
        self.netD.plot_layer_hists(show=show, save=save)

        self.troubleshoot_discriminator(show=show, save=save)
        self.troubleshoot_evaluator(real_netE=real_netE, show=show, save=save)

        cm_gen, cr_gen = self.netE.classification_stats(title='CGAN', show=show, save=save)

        print("\nCGAN Evaluator Network Classification Stats:\n")
        print(cm_gen)
        print("\n")
        print(cr_gen)

        cm_real, cr_real = real_netE.classification_stats(title='Real', show=show, save=save)

        print("\nReal Data Evaluator Network Classification Stats:\n")
        print(cm_real)
        print("\n")
        print(cr_real)

    def plot_progress(self, benchmark_acc, show, save=None):
        """
        Plot scores of each evaluation model across training of CGAN
        :param benchmark_acc: Best score obtained from training Evaluator on real data
        :param show: Whether to show the plot
        :param save: Where to save the plot. If set to None default path is used. If false, not saved.
        """
        if save is None:
            save = self.path

        length = len(self.stored_acc)

        plt.bar(x=range(length), height=self.stored_acc, tick_label=np.linspace(self.epoch // length, self.epoch, length, dtype=np.int64))
        plt.plot(np.linspace(0, length, length), np.full(length, benchmark_acc), linestyle='dashed', color='r')

        plt.xlabel('Evaluation', fontweight='bold')
        plt.ylabel('Accuracy (%)', fontweight='bold')
        plt.title('Evaluation Over Training Evaluations', fontweight='bold')

        if show:
            plt.show()

        if save:
            assert os.path.exists(save), "Check that the desired save path exists."
            plt.savefig(os.path.join(save, cs.FILENAME_PLOT_PROGRESS))

    def troubleshoot_discriminator(self, labels=None, num_examples=None, exit_early_iters=1000, gen=None, show=True, save=None):
        """
        Produce several nrow x nc grids of examples of interest for troubleshooting the model
        1. Grid of generated examples discriminator labeled as fake.
        2. Grid of generated examples discriminator labeled as real.
        3. Grid of real examples discriminator labeled as fake.
        4. Grid of real examples discriminator labeled as real.
        :param labels: Which classes to generate examples for. Default is all.
        :param num_examples: Number of examples of each class to generate examples for. Default is 10.
        :param exit_early_iters: Number of iterations to exit after if not enough images are found for grids 1 and 2
        :param gen: Generator to use for grids 3 and 4
        :param show: Whether to show the plots
        :param save: Where to save the plots. If set to None default path is used. If false, not saved.
        """
        if save is None:
            save = self.path

        if gen is None:
            gen = self.test_gen  # More data exists

        if num_examples is None:
            num_examples = self.grid_num_examples
        assert num_examples <= self.grid_num_examples, 'Num examples must be less than or equal to ' + str(self.grid_num_examples)

        if labels is None:
            labels = self.le.classes_

        grid1, grid2 = self.build_grid1_and_grid2(labels=labels, num_examples=num_examples, exit_early_iters=exit_early_iters)
        grid3, grid4 = self.build_grid3_and_grid4(labels=labels, num_examples=num_examples, gen=gen)

        grid1 = vutils.make_grid(tensor=grid1, nrow=num_examples, normalize=True).detach().cpu()
        grid2 = vutils.make_grid(tensor=grid2, nrow=num_examples, normalize=True).detach().cpu()
        grid3 = vutils.make_grid(tensor=grid3, nrow=num_examples, normalize=True).detach().cpu()
        grid4 = vutils.make_grid(tensor=grid4, nrow=num_examples, normalize=True).detach().cpu()

        f, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        axes[0, 0].title.set_text("Fake examples labeled as fake")
        axes[0, 1].title.set_text("Fake examples labeled as real")
        axes[1, 0].title.set_text("Real examples labeled as fake")
        axes[1, 1].title.set_text("Real examples labeled as real")

        axes[0, 0].imshow(np.transpose(grid1, (1, 2, 0)))
        axes[0, 1].imshow(np.transpose(grid2, (1, 2, 0)))
        axes[1, 0].imshow(np.transpose(grid3, (1, 2, 0)))
        axes[1, 1].imshow(np.transpose(grid4, (1, 2, 0)))

        st = f.suptitle("Troubleshooting examples of discriminator outputs", fontweight='bold', fontsize=20)
        f.tight_layout()
        st.set_y(0.96)
        f.subplots_adjust(top=0.9)

        if show:
            f.show()

        if save:
            assert os.path.exists(save), "Check that the desired save path exists."
            os.makedirs(os.path.join(save, 'troubleshoot_plots'), exist_ok=True)
            f.savefig(os.path.join(save, 'troubleshoot_plots', 'discriminator.png'))

    def troubleshoot_evaluator(self, real_netE, labels=None, num_examples=None, show=True, save=None):
        """
        Produce several nrow x nc grids of examples of interest for troubleshooting the model
        5. Grid of real examples that the evaluator failed to identify correctly (separate plot).
        6. Grid of what the evaluator THOUGHT each example in grid 5 should be.
        7. Grid of misclassified examples by model trained on real data.
        8. Grid of what the evaluator THOUGHT each example in grid 7 should be.
        :param real_netE: A version of netE trained on real data, rather than synthetic data
        :param labels: Which classes to generate examples for. Default is all.
        :param num_examples: Number of examples of each class to generate examples for. Default is 10.
        :param show: Whether to show the plots
        :param save: Where to save the plots. If set to None default path is used. If false, not saved.
        """
        if save is None:
            save = self.path

        if num_examples is None:
            num_examples = self.grid_num_examples
        assert num_examples <= self.grid_num_examples, 'Num examples must be less than or equal to ' + str(self.grid_num_examples)

        if labels is None:
            labels = self.le.classes_

        grid5, grid6 = self.build_eval_grids(netE=self.netE, labels=labels, num_examples=num_examples)
        grid7, grid8 = self.build_eval_grids(netE=real_netE, labels=labels, num_examples=num_examples)

        grid5 = vutils.make_grid(tensor=grid5, nrow=self.grid_num_examples, normalize=True).detach().cpu()
        grid6 = vutils.make_grid(tensor=grid6, nrow=self.grid_num_examples, normalize=True).detach().cpu()
        grid7 = vutils.make_grid(tensor=grid7, nrow=self.grid_num_examples, normalize=True).detach().cpu()
        grid8 = vutils.make_grid(tensor=grid8, nrow=self.grid_num_examples, normalize=True).detach().cpu()

        f, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        axes[0, 0].title.set_text("CGAN Eval Mistakes")
        axes[0, 1].title.set_text("CGAN Eval Intended")
        axes[1, 0].title.set_text("Real Data Eval Mistakes")
        axes[1, 1].title.set_text("Real Data Eval Intended")

        axes[0, 0].imshow(np.transpose(grid5, (1, 2, 0)))
        axes[0, 1].imshow(np.transpose(grid6, (1, 2, 0)))
        axes[1, 0].imshow(np.transpose(grid7, (1, 2, 0)))
        axes[1, 1].imshow(np.transpose(grid8, (1, 2, 0)))

        st = f.suptitle("Troubleshooting examples of evaluator outputs", fontweight='bold', fontsize=20)
        f.tight_layout()
        st.set_y(0.96)
        f.subplots_adjust(top=0.9)

        if show:
            f.show()

        if save:
            assert os.path.exists(save), "Check that the desired save path exists."
            os.makedirs(os.path.join(save, 'troubleshoot_plots'), exist_ok=True)
            f.savefig(os.path.join(save, 'troubleshoot_plots', 'evaluator.png'))

    def build_grid1_and_grid2(self, labels=None, num_examples=None, exit_early_iters=1000):
        """Generate images and feeds them to discriminator in order to find num_examples of each class specified"""
        if num_examples is None:
            num_examples = self.grid_num_examples
        assert num_examples <= self.grid_num_examples, 'Num examples must be less than or equal to ' + str(self.grid_num_examples)

        if labels is None:
            labels = self.le.classes_

        self.netG.eval()
        self.netD.eval()
        bs = self.fake_bs  # Seems to be a good number with training above.

        grid1 = torch.zeros(num_examples * len(labels), self.num_channels, self.x_dim[0], self.x_dim[1])
        grid2 = torch.zeros(num_examples * len(labels), self.num_channels, self.x_dim[0], self.x_dim[1])

        grid1_counts = {}  # Represents the number of each class acquired so far for this grid
        grid2_counts = {}

        for i in range(len(labels)):
            grid1_counts[i] = 0
            grid2_counts[i] = 0

        count = 0

        while not (all(x == num_examples for x in grid1_counts.values()) and all(x == num_examples for x in grid2_counts.values())) and count < exit_early_iters:
            noise = torch.randn(bs, self.nz, device=self.device)
            random_labels = iu.convert_y_to_one_hot(y=torch.from_numpy(np.random.choice(self.le.transform(labels), bs)), nc=self.nc).to(self.device).type(torch.float32)

            with torch.no_grad():
                fakes = self.netG(noise, random_labels)
                fwd = self.netD(fakes, random_labels)

            for i in range(len(labels)):
                grid1_contenders = fakes[(random_labels[:, self.le.transform(labels)[i]] == 1) * (fwd[:, 0] < 0.5)]
                grid2_contenders = fakes[(random_labels[:, self.le.transform(labels)[i]] == 1) * (fwd[:, 0] > 0.5)]

                grid1_retain = min(num_examples - grid1_counts[i], len(grid1_contenders))
                grid2_retain = min(num_examples - grid2_counts[i], len(grid2_contenders))

                grid1[(i * num_examples) + grid1_counts[i]:(i * num_examples) + grid1_counts[i] + grid1_retain] = grid1_contenders[:grid1_retain]
                grid2[(i * num_examples) + grid2_counts[i]:(i * num_examples) + grid2_counts[i] + grid2_retain] = grid2_contenders[:grid2_retain]

                grid1_counts[i] += grid1_retain
                grid2_counts[i] += grid2_retain

            count += 1

        return grid1, grid2

    def build_grid3_and_grid4(self, gen, labels=None, num_examples=None):
        """
        Feed real images to discriminator in order to find num_examples of each specified class labeled as fake
        Runs one full epoch over training data
        """
        if num_examples is None:
            num_examples = self.grid_num_examples
        assert num_examples <= self.grid_num_examples, 'Num examples must be less than or equal to ' + str(self.grid_num_examples)

        if labels is None:
            labels = self.le.classes_

        self.netD.eval()

        grid3 = torch.zeros(num_examples * len(labels), self.num_channels, self.x_dim[0], self.x_dim[1])
        grid4 = torch.zeros(num_examples * len(labels), self.num_channels, self.x_dim[0], self.x_dim[1])

        grid3_counts = {}  # Represents the number of each class acquired so far for this grid
        grid4_counts = {}

        for i in range(len(labels)):
            grid3_counts[i] = 0
            grid4_counts[i] = 0

        for x, y in gen:
            x, y = x.to(self.device), y.type(torch.float32).to(self.device)
            y = iu.convert_y_to_one_hot(y=y.type(torch.LongTensor).cpu().detach(), nc=self.nc).to(self.device).type(torch.float32)

            with torch.no_grad():
                fwd = self.netD(x, y)

            for i in range(len(labels)):
                grid3_contenders = x[(y[:, self.le.transform(labels)[i]] == 1) * (fwd[:, 0] < 0.5)]
                grid4_contenders = x[(y[:, self.le.transform(labels)[i]] == 1) * (fwd[:, 0] > 0.5)]

                grid3_retain = min(num_examples - grid3_counts[i], len(grid3_contenders))
                grid4_retain = min(num_examples - grid4_counts[i], len(grid4_contenders))

                grid3[(i * num_examples) + grid3_counts[i]:(i * num_examples) + grid3_counts[i] + grid3_retain] = grid3_contenders[:grid3_retain]
                grid4[(i * num_examples) + grid4_counts[i]:(i * num_examples) + grid4_counts[i] + grid4_retain] = grid4_contenders[:grid4_retain]

                grid3_counts[i] += grid3_retain
                grid4_counts[i] += grid4_retain

                # Exit early if grid filled up
                if all(x == num_examples for x in grid3_counts.values()) and all(x == num_examples for x in grid4_counts.values()):
                    return grid3, grid4

        return grid3, grid4

    def build_eval_grids(self, netE, labels=None, num_examples=None):
        """Construct grids 5-8 for troubleshoot_evaluator method"""
        if num_examples is None:
            num_examples = self.grid_num_examples
        assert num_examples <= self.grid_num_examples, 'Num examples must be less than or equal to ' + str(self.grid_num_examples)

        if labels is None:
            labels = self.le.classes_

        netE.eval()

        grid1 = torch.zeros(num_examples * len(labels), self.num_channels, self.x_dim[0], self.x_dim[1])
        grid2 = torch.zeros(num_examples * len(labels), self.num_channels, self.x_dim[0], self.x_dim[1])

        grid1_counts = {}  # Represents the number of each class acquired so far for this grid

        for i in range(num_examples):
            grid1_counts[i] = 0

        for x, y in self.test_gen:
            x, y = x.to(self.device), y.type(torch.float32).to(self.device)
            y = iu.convert_y_to_one_hot(y=y.type(torch.LongTensor).cpu().detach(), nc=self.nc).to(self.device).type(torch.float32)

            with torch.no_grad():
                fwd = netE(x)

            for i in range(len(labels)):
                grid1_contenders = x[(torch.argmax(y, -1) != torch.argmax(fwd, -1)) * (torch.argmax(y, -1) == self.le.transform(labels)[i])]

                if len(grid1_contenders) > 0:
                    grid1_intended = torch.argmax(fwd[(torch.argmax(y, -1) != torch.argmax(fwd, -1)) * (torch.argmax(y, -1) == self.le.transform(labels)[i])], -1)

                    grid2_contenders = torch.zeros(0, self.num_channels, self.x_dim[0], self.x_dim[1]).to(self.device)
                    for mistake in grid1_intended:
                        img = self.find_particular_img(gen=self.train_gen, net=None, mistake=None, label=self.le.inverse_transform([mistake.item()])[0])
                        if img is None:
                            img = torch.zeros((1, self.num_channels, self.x_dim[0], self.x_dim[1]), dtype=torch.float32).to(self.device)
                        else:
                            img = img.view(-1, self.num_channels, self.x_dim[0], self.x_dim[1])
                        grid2_contenders = torch.cat((grid2_contenders, img), dim=0)

                    grid1_retain = min(num_examples - grid1_counts[i], len(grid1_contenders))

                    grid1[(i * num_examples) + grid1_counts[i]:(i * num_examples) + grid1_counts[i] + grid1_retain] = grid1_contenders[:grid1_retain]
                    grid2[(i * num_examples) + grid1_counts[i]:(i * num_examples) + grid1_counts[i] + grid1_retain] = grid2_contenders[:grid1_retain]

                    grid1_counts[i] += grid1_retain

                # Exit early if grid filled up
                if all(x == num_examples for x in grid1_counts.values()):
                    return grid1, grid2

        return grid1, grid2

    def find_particular_img(self, gen, net, label, mistake, escape=999):
        """
        Searches through the generator to find a single image of interest based on search parameters
        :param gen: Generator to use. netG is a valid generator to use for fake data.
        :param net: Network to use. Either netD or netE.
        :param label: Label to return (0-9)
        :param mistake: Whether the example should be a mistake (True or False)
        :param escape: Error out if loop hits this number
        :return: torch tensor of image (x_dim[0] x x_dim[1])
        """
        assert gen in {self.train_gen, self.val_gen, self.test_gen, self.netG}, "Please use a valid generator (train/val/test/generator)"
        assert (mistake is None) == (net is None), "Either both mistake and net must be None, or neither must be None"
        assert net in {self.netD, self.netE, None}, "Please use a valid net (netD, netE, or None)"
        assert mistake in {True, False, None}, "Mistake should be True, False, or None"
        assert label in self.le.classes_, "Make sure label is a valid class"

        label = self.le.transform([label]).take(0)

        bs = self.fake_bs

        if net is not None:
            net.eval()

        escape_counter = 0

        while True:  # Search until a match is found
            escape_counter += 1

            # Generate examples
            if gen == self.netG:
                noise = torch.randn(bs, self.nz, device=self.device)
                y = iu.convert_y_to_one_hot(y=torch.full((bs, 1), label, dtype=torch.int64), nc=self.nc).to(self.device).type(torch.float32)

                with torch.no_grad():
                    x = self.netG(noise, y)

            else:
                iterator = gen.__iter__()
                x, y = next(iterator)
                x, y = x.to(self.device), y.type(torch.float32).to(self.device)
                boolz = y == label
                x, y = x[boolz], y[boolz]

            if len(x) > 0:
                if mistake is None:
                    return x[0]
                y = iu.convert_y_to_one_hot(y=y.type(torch.LongTensor).cpu().detach(), nc=self.nc).to(self.device).type(torch.float32)
                with torch.no_grad():
                    if net == self.netD:
                        fwd = net(x, y)
                    else:
                        fwd = net(x)

                # Check if conditions are met and exit, otherwise continue.
                # netD and incorrect
                if net == self.netD:
                    if mistake:
                        if gen == self.netG:  # Incorrect means classifying as real
                            contenders = x[fwd > 0.5]
                        else:
                            contenders = x[fwd < 0.5]
                    # netD and correct
                    else:
                        if gen == self.netG:  # Correct means classifying as fake
                            contenders = x[fwd < 0.5]
                        else:
                            contenders = x[fwd > 0.5]
                # netE and incorrect
                elif mistake:
                    contenders = x[torch.argmax(fwd, -1) != torch.argmax(y, -1)]
                # netE and incorrect
                else:
                    contenders = x[torch.argmax(fwd, -1) == torch.argmax(y, -1)]

                # If 1 or more values returned, return that value and exit. Otherwise, continue.
                if len(contenders) > 0:
                    return contenders[0]

                if escape_counter == escape:
                    return None

    def draw_cam(self, gen, net, label, mistake, show, path, scale=None, escape=999):
        """
        Wrapper function for find_particular_img and draw_cam
        :param gen: Generator to use. netG is a valid generator to use for fake data (otherwise data_gen).
        :param net: Network to use. Either netD or netE.
        :param label: Label to return
        :param mistake: Whether the example should be a mistake (True or False)
        :param show: Whether to show the image
        :param path: Path to create image file. Needs full file name. Should end in .jpg
        :param scale: Multiplier to scale image back to original values
        :param escape: Error out if loop hits this number
        """
        assert path.split(".")[-1] == "jpg", "Please make sure path ends in '.jpg'"
        assert label in self.le.classes_, "Make sure label is a valid class"

        if scale is None:
            scale = 1 if self.num_channels > 1 else 255

        img = self.find_particular_img(gen=gen, net=net, label=label, mistake=mistake, escape=escape)

        assert img is not None, "Unable to find an image within the specified timeframe"

        real = gen != self.netG

        if net == self.netD:
            label = self.le.transform([label])[0]
            self.netD.draw_cam(img=img, label=label, path=path, scale=scale, show=show, real=real)
        else:
            self.netE.draw_cam(img=img, path=path, scale=scale, show=show, real=real)

    def extract_x_dim(self):
        iterator = iter(self.train_gen)
        x, _, = next(iterator)
        return x.shape[-2], x.shape[-1]

    def gen_data(self, size, path, stratify=None, label=None):
        """Generates a data set formatted like the original data and saves to specified path"""
        assert os.path.exists(path), "Output directory exists"

        bs = min(self.fake_bs, size)

        dataset = OnlineGeneratedImageDataset(netG=self.netG, size=size, nz=self.nz, nc=self.nc, bs=bs,
                                              ohe=self.ohe, device=self.device, x_dim=self.x_dim, stratify=stratify)
        gen = data.DataLoader(dataset, batch_size=bs,
                              shuffle=False, num_workers=self.fake_num_workers)

        label = 'genned_img' if label is None else label

        gen.dataset.next_epoch()
        with torch.no_grad():
            for i in range(gen.dataset.batches_per_epoch):
                batch, labels = gen.dataset.next_batch()
                for j, img in enumerate(batch):
                    vutils.save_image(img, os.path.join(path, label + '_' + str(i + j) + '.png'))
