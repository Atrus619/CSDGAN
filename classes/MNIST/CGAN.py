from classes.MNIST.MNIST_Dataset import Fake_MNIST_Dataset
from torch.utils import data
from classes.MNIST.netD import CGAN_Discriminator
from classes.MNIST.netG import CGAN_Generator
from classes.MNIST.netE import CGAN_Evaluator
from classes.NetUtils import GaussianNoise
from utils.MNIST import *
import time
from utils.utils import *
import imageio
from torchviz import make_dot


# This CGAN will be set up a bit differently in the hopes of being cleaner. I am going to enclose netG and netD into a higher level class titled CGAN.
class CGAN(nn.Module):
    def __init__(self, train_gen, val_gen, test_gen, device, x_dim, nc, nz, num_channels, sched_netG, path,
                 label_noise, label_noise_linear_anneal, discrim_noise, discrim_noise_linear_anneal,
                 netG_nf, netG_lr, netG_beta1, netG_beta2, netG_wd,
                 netD_nf, netD_lr, netD_beta1, netD_beta2, netD_wd,
                 netE_lr, netE_beta1, netE_beta2, netE_wd,
                 fake_data_set_size, fake_bs,
                 eval_num_epochs, early_stopping_patience):
        # Inherit nn.Module initialization
        super(CGAN, self).__init__()

        self.path = path  # default file path for saved objects
        safe_mkdir(self.path)
        safe_mkdir(self.path + "/stored_generators")

        self.sched_netG = sched_netG

        assert 0.0 <= label_noise <= 1.0, "Label noise must be between 0 and 1"
        self.label_noise = label_noise
        self.label_noise_linear_anneal = label_noise_linear_anneal
        self.ln_rate = 0.0

        self.discrim_noise = discrim_noise
        self.discrim_noise_linear_anneal = discrim_noise_linear_anneal
        self.dn_rate = 0.0

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

        self.real_label = 1
        self.fake_label = 0

        self.epoch = 0

        self.stored_loss = []
        self.stored_acc = []

        # Instantiate sub-nets
        self.netG = CGAN_Generator(nz=self.nz, num_channels=self.num_channels, nf=netG_nf, x_dim=self.x_dim, nc=self.nc, device=self.device, path=self.path,
                                   lr=netG_lr, beta1=netG_beta1, beta2=netG_beta2, wd=netG_wd).to(self.device)
        self.netD = CGAN_Discriminator(nf=netD_nf, num_channels=self.num_channels, nc=self.nc, noise=self.discrim_noise, device=self.device, x_dim=self.x_dim, path=self.path,
                                       lr=netD_lr, beta1=netD_beta1, beta2=netD_beta2, wd=netD_wd).to(self.device)

        self.netE_params = {'lr': netE_lr, 'beta1': netE_beta1, 'beta2': netE_beta2, 'wd': netE_wd}
        self.netE = None  # Initialized through init_evaluator method
        self.eval_num_epochs = eval_num_epochs
        self.early_stopping_patience = early_stopping_patience

        self.fixed_imgs = [self.gen_fixed_img_grid()]

    def train_gan(self, num_epochs, print_freq, eval_freq=None):
        """
        Primary method for training
        :param num_epochs: Desired number of epochs to train for
        :param print_freq: How freqently to print out training statistics (i.e., freq of 5 will result in information being printed every 5 epochs)
        :param eval_freq: How frequently to evaluate with netE. If None, no evaluation will occur. Evaluation takes a significant amount of time.
        """
        total_epochs = self.epoch + num_epochs

        if self.label_noise_linear_anneal:
            self.ln_rate = self.label_noise / num_epochs

        if self.discrim_noise_linear_anneal:
            self.dn_rate = self.discrim_noise / num_epochs

        print("Beginning training")
        start_time = time.time()
        for epoch in range(num_epochs):
            for x, y in self.train_gen:
                x, y = x.to(self.device), y.to(self.device)
                self.train_one_step(x, y)

            self.next_epoch()

            if self.epoch % print_freq == 0 or (self.epoch == num_epochs):
                print("Time: %ds" % (time.time() - start_time))
                start_time = time.time()

                self.print_progress(total_epochs)

            if eval_freq is not None:
                if self.epoch % eval_freq == 0 or (self.epoch == num_epochs):
                    self.init_fake_gen()
                    self.test_model(train_gen=self.fake_train_gen, val_gen=self.fake_val_gen)
                    print("Epoch: %d\tEvaluator Score: %.4f" % (self.epoch, self.stored_acc[-1]))

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

    def test_model(self, train_gen, val_gen):
        """
        Train a CNN evaluator from scratch
        :param train_gen: Specified train_gen, can either be real training generator or a created one from netG
        :param val_gen: Same as above ^
        """
        self.init_evaluator(train_gen, val_gen)
        self.netE.train_evaluator(num_epochs=self.eval_num_epochs, eval_freq=1, es=self.early_stopping_patience)
        torch.save(self.netG.state_dict, self.path + "/stored_generators/Epoch_" + str(self.epoch) + "_Generator.pt")
        loss, acc = self.netE.eval_once(self.test_gen)
        self.stored_loss.append(loss.item())
        self.stored_acc.append(acc.item())

    def print_progress(self, total_epochs):
        """Print metrics of interest"""
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (self.epoch, total_epochs, self.netD.losses[-1], self.netG.losses[-1], self.netD.Avg_D_reals[-1], self.netD.Avg_D_fakes[-1], self.netG.Avg_G_fakes[-1]))

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
        self.netE = CGAN_Evaluator(train_gen=train_gen, val_gen=val_gen, test_gen=self.test_gen, device=self.device, x_dim=self.x_dim, num_channels=self.num_channels,
                                   nc=self.nc, path=self.path, **self.netE_params).to(self.device)

    def init_fake_gen(self):
        # Initialize fake training set and validation set to be same size
        self.fake_train_set = Fake_MNIST_Dataset(self.netG, self.fake_data_set_size, self.nz, self.nc, self.device)
        self.fake_train_gen = data.DataLoader(self.fake_train_set, batch_size=self.fake_bs, shuffle=self.fake_shuffle, num_workers=self.fake_num_workers)

        self.fake_val_set = Fake_MNIST_Dataset(self.netG, self.fake_data_set_size, self.nz, self.nc, self.device)
        self.fake_val_gen = data.DataLoader(self.fake_val_set, batch_size=self.fake_bs, shuffle=self.fake_shuffle, num_workers=self.fake_num_workers)

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
        self.netE.train_evaluator(num_epochs=num_epochs, eval_freq=1, es=es)
        _, og_result = self.netE.eval_once(test_gen)
        og_result = og_result.numpy().take(0)
        return og_result, copy.copy(self.netE)

    def show_img(self, label):
        """Generate a 28x28 image based on the desired class label index (integer 0-9)"""
        assert 0 <= label <= 9 and type(label) is int, "Make sure label is an integer between 0 and 9 (inclusive)."
        noise = torch.randn(1, self.nz, device=self.device)
        processed_label = torch.zeros([1, 10], dtype=torch.uint8, device='cpu')
        label = torch.full((1, 1), label, dtype=torch.int64)
        processed_label = processed_label.scatter(1, label, 1).float().to(self.device)
        output = self.netG(noise, processed_label).view(28, 28).detach().cpu().numpy()
        plt.imshow(output, cmap='gray')
        plt.show()

    def gen_fixed_img_grid(self):
        """
        Produce a grid of generated images from netG's fixed noise vector. This can be used to visually track progress of the CGAN training.
        :return: Tensor of images
        """
        self.netG.eval()
        with torch.no_grad():
            fixed_imgs = self.netG(self.netG.fixed_noise, self.netG.fixed_labels)
        return vutils.make_grid(tensor=fixed_imgs, nrow=10, normalize=True).detach().cpu()

    def show_grid(self, index):
        """
        Print a specified fixed image grid from the self.fixed_imgs list
        :param index: Evaluation index to display
        :return: Nothing. Displays the desired image instead.
        """
        assert len(self.fixed_imgs) > 0, "Model not yet trained"
        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(np.transpose(self.fixed_imgs[index], (1, 2, 0)))
        plt.show()

    def build_gif(self, path=None):
        """
        Loop through self.fixed_imgs and saves the images to a folder.
        :param path: Path to folder to save images. Folder will be created if it does not already exist.
        """
        assert len(self.fixed_imgs) > 0, "Model not yet trained"

        if path is None:
            path = self.path

        safe_mkdir(path + "/imgs")
        ims = []
        for epoch, grid in enumerate(self.fixed_imgs):
            fig = plt.figure(figsize=(8, 8))
            plt.axis('off')
            plt.suptitle('Epoch ' + str(epoch))
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            img_name = path + "/imgs/Epoch " + str(epoch) + ".png"
            plt.savefig(img_name)
            ims.append(imageio.imread(img_name))
            plt.close()
            if epoch == self.epoch:  # Hacky method to stay on the final frame for longer
                for i in range(20):
                    ims.append(imageio.imread(img_name))
                    plt.close()
        imageio.mimsave(path + '/generation_animation.gif', ims, fps=5)

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
        self.netG.build_hist_gif(path=save, net="Generator")
        self.netD.build_hist_gif(path=save, net="Discriminator")

        self.plot_training_plots(show=show, save=save)

        self.netG.plot_layer_scatters(title="Generator", show=show, save=save)
        self.netD.plot_layer_scatters(title="Discriminator", show=show, save=save)

        self.netG.plot_layer_hists(title="Generator", show=show, save=save)
        self.netD.plot_layer_hists(title="Discriminator", show=show, save=save)

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

        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Accuracy (%)', fontweight='bold')
        plt.title('Evaluation Over Training Epochs', fontweight='bold')

        if show:
            plt.show()

        if save:
            assert os.path.exists(save), "Check that the desired save path exists."
            plt.savefig(save + '/training_progress.png')

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

    def load_netG(self, epoch):
        """Load a previously stored netG (likely the one that performed the best)"""
        self.netG.load_state_dict(torch.load(self.path + "/stored_generators/Epoch_" + epoch + "_Generator.pt"))

    def troubleshoot_discriminator(self, exit_early_iters=1000, gen=None, show=True, save=None):
        """
        Produce several 10x10 grids of examples of interest for troubleshooting the model
        1. 10x10 grid of generated examples discriminator labeled as fake.
        2. 10x10 grid of generated examples discriminator labeled as real.
        3. 10x10 grid of real examples discriminator labeled as fake.
        4. 10x10 grid of real examples discriminator labeled as real.
        :param exit_early_iters: Number of iterations to exit after if not enough images are found for grids 1 and 2
        :param gen: Generator to use for grids 3 and 4
        :param show: Whether to show the plots
        :param save: Where to save the plots. If set to None default path is used. If false, not saved.
        """
        if save is None:
            save = self.path

        if gen is None:
            gen = self.test_gen  # More data exists

        grid1, grid2 = self.build_grid1_and_grid2(exit_early_iters=exit_early_iters)
        grid3, grid4 = self.build_grid3_and_grid4(gen=gen)

        grid1, grid2 = vutils.make_grid(tensor=grid1, nrow=10, normalize=True).detach().cpu(), vutils.make_grid(tensor=grid2, nrow=10, normalize=True).detach().cpu()
        grid3, grid4 = vutils.make_grid(tensor=grid3, nrow=10, normalize=True).detach().cpu(), vutils.make_grid(tensor=grid4, nrow=10, normalize=True).detach().cpu()

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
            safe_mkdir(save + '/troubleshoot_plots')
            f.savefig(save + '/troubleshoot_plots/discriminator.png')

    def troubleshoot_evaluator(self, real_netE, show=True, save=None):
        """
        Produce several 10x10 grids of examples of interest for troubleshooting the model
        5. 10x10 grid of real examples that the evaluator failed to identify correctly (separate plot).
        6. 10x10 grid of what the evaluator THOUGHT each example in grid 5 should be.
        7. 10x10 grid of misclassified examples by model trained on real data.
        8. 10x10 grid of what the evaluator THOUGHT each example in grid 7 should be.
        :param show: Whether to show the plots
        :param save: Where to save the plots. If set to None default path is used. If false, not saved.
        """
        if save is None:
            save = self.path

        grid5, grid6 = self.build_eval_grids(netE=self.netE)
        grid7, grid8 = self.build_eval_grids(netE=real_netE)

        grid5, grid6 = vutils.make_grid(tensor=grid5, nrow=10, normalize=True).detach().cpu(), vutils.make_grid(tensor=grid6, nrow=10, normalize=True).detach().cpu()
        grid7, grid8 = vutils.make_grid(tensor=grid7, nrow=10, normalize=True).detach().cpu(), vutils.make_grid(tensor=grid8, nrow=10, normalize=True).detach().cpu()

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
            safe_mkdir(save + '/troubleshoot_plots')
            f.savefig(save + '/troubleshoot_plots/evaluator.png')

    def build_grid1_and_grid2(self, exit_early_iters=1000):
        """Generate images and feeds them to discriminator in order to find 10 examples of each class"""
        self.netG.eval()
        self.netD.eval()
        bs = 128  # Seems to be a good number with training above.

        grid1 = torch.zeros(100, self.num_channels, self.x_dim[0], self.x_dim[1])
        grid2 = torch.zeros(100, self.num_channels, self.x_dim[0], self.x_dim[1])

        grid1_counts = {}  # Represents the number of each class acquired so far for this grid
        grid2_counts = {}

        for i in range(10):
            grid1_counts[i] = 0
            grid2_counts[i] = 0

        count = 0

        while not (all(x == 10 for x in grid1_counts.values()) and all(x == 10 for x in grid2_counts.values())) and count < exit_early_iters:
            noise = torch.randn(bs, self.nz, device=self.device)
            random_labels = convert_y_to_one_hot(torch.from_numpy(np.random.randint(0, 10, bs))).to(self.device).type(torch.float32)

            with torch.no_grad():
                fakes = self.netG(noise, random_labels)
                fwd = self.netD(fakes, random_labels)

            for i in range(10):
                grid1_contenders = fakes[(random_labels[:, i] == 1) * (fwd[:, 0] < 0.5)]
                grid2_contenders = fakes[(random_labels[:, i] == 1) * (fwd[:, 0] > 0.5)]

                grid1_retain = min(10 - grid1_counts[i], len(grid1_contenders))
                grid2_retain = min(10 - grid2_counts[i], len(grid2_contenders))

                grid1[(i*10) + grid1_counts[i]:(i*10) + grid1_counts[i]+grid1_retain] = grid1_contenders[:grid1_retain]
                grid2[(i*10) + grid2_counts[i]:(i*10) + grid2_counts[i]+grid2_retain] = grid2_contenders[:grid2_retain]

                grid1_counts[i] += grid1_retain
                grid2_counts[i] += grid2_retain

            count += 1

        return grid1, grid2

    def build_grid3_and_grid4(self, gen):
        """
        Feed real images to discriminator in order to find 10 examples of each class labeled as fake
        Runs one full epoch over training data
        """
        self.netD.eval()

        grid3 = torch.zeros(100, self.num_channels, self.x_dim[0], self.x_dim[1])
        grid4 = torch.zeros(100, self.num_channels, self.x_dim[0], self.x_dim[1])

        grid3_counts = {}  # Represents the number of each class acquired so far for this grid
        grid4_counts = {}

        for i in range(10):
            grid3_counts[i] = 0
            grid4_counts[i] = 0

        for x, y in gen:
            x, y = x.to(self.device), y.type(torch.float32).to(self.device)

            with torch.no_grad():
                fwd = self.netD(x, y)

            for i in range(10):
                grid3_contenders = x[(y[:, i] == 1) * (fwd[:, 0] < 0.5)]
                grid4_contenders = x[(y[:, i] == 1) * (fwd[:, 0] > 0.5)]

                grid3_retain = min(10 - grid3_counts[i], len(grid3_contenders))
                grid4_retain = min(10 - grid4_counts[i], len(grid4_contenders))

                grid3[(i*10) + grid3_counts[i]:(i*10) + grid3_counts[i]+grid3_retain] = grid3_contenders[:grid3_retain]
                grid4[(i*10) + grid4_counts[i]:(i*10) + grid4_counts[i]+grid4_retain] = grid4_contenders[:grid4_retain]

                grid3_counts[i] += grid3_retain
                grid4_counts[i] += grid4_retain

                # Exit early if grid filled up
                if all(x == 10 for x in grid3_counts.values()) and all(x == 10 for x in grid4_counts.values()):
                    return grid3, grid4

        return grid3, grid4

    def build_eval_grids(self, netE):
        """Construct grids 5-8 for troubleshoot_evaluator method"""
        netE.eval()

        grid1 = torch.zeros(100, self.num_channels, self.x_dim[0], self.x_dim[1])
        grid2 = torch.zeros(100, self.num_channels, self.x_dim[0], self.x_dim[1])

        grid1_counts = {}  # Represents the number of each class acquired so far for this grid

        for i in range(10):
            grid1_counts[i] = 0

        for x, y in self.test_gen:
            x, y = x.to(self.device), y.type(torch.float32).to(self.device)

            with torch.no_grad():
                fwd = netE(x)

            for i in range(10):
                grid1_contenders = x[(torch.argmax(y, -1) != torch.argmax(fwd, -1)) * (torch.argmax(y, -1) == i)]

                if len(grid1_contenders) > 0:
                    grid1_intended = torch.argmax(fwd[(torch.argmax(y, -1) != torch.argmax(fwd, -1)) * (torch.argmax(y, -1) == i)], -1)

                    grid2_contenders = torch.zeros(0, self.num_channels, self.x_dim[0], self.x_dim[1]).to(self.device)
                    for mistake in grid1_intended:
                        grid2_contenders = torch.cat((grid2_contenders, x[torch.argmax(y, -1) == mistake][0].view(-1, self.num_channels, self.x_dim[0], self.x_dim[1])), dim=0)

                    grid1_retain = min(10 - grid1_counts[i], len(grid1_contenders))

                    grid1[(i*10) + grid1_counts[i]:(i*10) + grid1_counts[i]+grid1_retain] = grid1_contenders[:grid1_retain]
                    grid2[(i*10) + grid1_counts[i]:(i*10) + grid1_counts[i]+grid1_retain] = grid2_contenders[:grid1_retain]

                    grid1_counts[i] += grid1_retain

                # Exit early if grid filled up
                if all(x == 10 for x in grid1_counts.values()):
                    return grid1, grid2

        return grid1, grid2

    def find_particular_img(self, gen, net, label, mistake):
        """
        Searches through the generator to find a single image of interest based on search parameters
        :param gen: Generator to use. netG is a valid generator to use for fake data.
        :param net: Network to use. Either "D" or "E".
        :param label: Label to return (0-9)
        :param mistake: Whether the example should be a mistake (True or False)
        :return: torch tensor of image (x_dim[0] x x_dim[1])
        """
        assert gen in {self.train_gen, self.val_gen, self.test_gen, self.netG}, "Please use a valid generator (train/val/test/generator)"
        assert net in {"D", "E"}, "Please use a valid net ('D' or 'E')"
        assert label in range(self.nc), "Please use a valid label #"
        assert mistake in {True, False}, "Mistake should be True or False"

        bs = 128

        subnet = self.netD if net == "D" else self.netE
        subnet.eval()

        while True:  # Search until a match is found
            # Generate examples
            if gen == self.netG:
                noise = torch.randn(bs, self.nz, device=self.device)
                y = convert_y_to_one_hot(torch.full((bs, 1), label, dtype=torch.int64)).to(self.device).type(torch.float32)

                with torch.no_grad():
                    x = self.netG(noise, y)

            else:
                iterator = gen.__iter__()
                x, y = next(iterator)
                x, y = x.to(self.device), y.type(torch.float32).to(self.device)
                intended_y = convert_y_to_one_hot(torch.full((bs, 1), label, dtype=torch.int64)).to(self.device).type(torch.float32)
                boolz = torch.argmax(y, -1) == torch.argmax(intended_y, -1)
                x, y = x[boolz], y[boolz]

            with torch.no_grad():
                if net == "D":
                    fwd = subnet(x, y)
                else:
                    fwd = subnet(x)

            # Check if conditions are met and exit, otherwise continue.
            # netD and incorrect
            if net == "D":
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

    def draw_cam(self, gen, net, label, mistake, show, path):
        """
        Wrapper function for find_particular_img and draw_cam
        :param gen: Generator to use. netG is a valid generator to use for fake data.
        :param net: Network to use. Either "D" or "E".
        :param label: Label to return (0-9)
        :param mistake: Whether the example should be a mistake (True or False)
        :param show: Whether to show the image
        :param path: Path to create image file. Needs full file name. Should end in .jpg
        """
        assert path.split(".")[-1] == "jpg", "Please make sure path ends in '.jpg'"

        img = self.find_particular_img(gen=gen, net=net, label=label, mistake=mistake)

        real = gen != self.netG

        if net == "D":
            self.netD.draw_cam(img=img, label=label, path=path, show=show, real=real)
        else:
            self.netE.draw_cam(img=img, path=path, show=show, real=real)

    def draw_architecture(self, net, show, save):
        """
        Utilizes torchviz to print current graph to a pdf
        :param net: Network to draw graph for. One of netG, netD, or netE.
        :param show: Whether to show the graph. To visualize in jupyter notebooks, run the returned viz.
        :param save: Where to save the graph.
        """
        assert net in {self.netG, self.netD, self.netE}, "Invalid entry for net. Should be netG, netD, or netE."

        if save is None:
            save = self.path

        iterator = self.train_gen.__iter__()
        x, y = next(iterator)
        x, y = x.to(self.device), y.to(self.device).type(torch.float32)

        if net == self.netG:
            noise = torch.randn(x.shape[0], self.nz, device=self.device)
            viz = make_dot(net(noise, y), params=dict(net.named_parameters()))
        elif net == self.netD:
            viz = make_dot(net(x, y), params=dict(net.named_parameters()))
        else:
            viz = make_dot(net(x), params=dict(net.named_parameters()))

        if net == self.netG:
            title = "Generator"
        elif net == self.netD:
            title = "Discriminator"
        else:
            title = "Evaluator"

        safe_mkdir(save + "/architectures")
        viz.render(filename=save + "/architectures/" + title, view=show)

        return viz
