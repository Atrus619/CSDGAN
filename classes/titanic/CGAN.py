from classes.titanic.netG import netG
from classes.titanic.netD import netD
from utils.utils import *
import time
from classes.NetUtils import GaussianNoise
import numpy as np
import shutil
import re
from torchviz import make_dot


class CGAN:
    def __init__(self, device, nc, nz, sched_netG, path, data_gen,
                 netG_H, netG_lr, netG_beta1, netG_beta2, netG_wd,
                 netD_H, netD_lr, netD_beta1, netD_beta2, netD_wd,
                 eval_param_grid, eval_folds, test_ranges, seed, eval_stratify,
                 label_noise, label_noise_linear_anneal, discrim_noise, discrim_noise_linear_anneal):
        self.path = path  # default file path for saved objects
        safe_mkdir(self.path)

        # Empty and rebuild stored generator directory for each CGAN
        stored_gen_path = os.path.join(self.path, "stored_generators")
        if os.path.exists(stored_gen_path):
            shutil.rmtree(stored_gen_path)
        safe_mkdir(stored_gen_path)

        # Initialize properties
        self.device = device
        self.nc = nc
        self.nz = nz
        self.data_gen = data_gen
        self.labels_list = self.data_gen.dataset.labels_list
        self.out_dim = self.data_gen.dataset.out_dim

        # Evaluator properties
        self.eval_param_grid = eval_param_grid
        self.eval_folds = eval_folds
        self.test_ranges = test_ranges
        self.seed = seed
        self.eval_stratify = eval_stratify

        # Anti-discriminator properties
        assert 0.0 <= label_noise <= 1.0, "Label noise must be between 0 and 1"
        self.label_noise = label_noise
        self.label_noise_linear_anneal = label_noise_linear_anneal
        self.ln_rate = 0.0

        self.discrim_noise = discrim_noise
        self.discrim_noise_linear_anneal = discrim_noise_linear_anneal
        self.dn_rate = 0.0

        # Instantiate sub-nets
        self.netG = netG(nz=self.nz, H=netG_H, out_dim=self.out_dim, nc=self.nc, device=self.device,
                         wd=netG_wd, cat_mask=self.data_gen.dataset.preprocessed_cat_mask, le_dict=self.data_gen.dataset.le_dict,
                         lr=netG_lr, beta1=netG_beta1, beta2=netG_beta2).to(device)
        self.netD = netD(H=netD_H, out_dim=self.out_dim, nc=self.nc, device=self.device, wd=netD_wd, noise = self.discrim_noise,
                         lr=netD_lr, beta1=netD_beta1, beta2=netD_beta2).to(device)

        # Training properties
        self.epoch = 0
        self.sched_netG = sched_netG
        self.real_label = 1
        self.fake_label = 0
        self.stored_acc = []

    def train_gan(self, num_epochs, cadence, print_freq, eval_freq=None):
        """
        Primary method for training
        :param num_epochs: Desired number of epochs to train for
        :param cadence: Number of pass-throughs of data set per epoch (generally set to 1, might want to set higher for very tiny data sets)
        :param print_freq: How freqently to print out training statistics (i.e., freq of 5 will result in information being printed every 5 epochs)
        :param eval_freq: How frequently to evaluate with netE. If None, no evaluation will occur.
        """
        total_epochs = self.epoch + num_epochs
        device_check = self.data_gen.dataset.device != self.device

        if self.label_noise_linear_anneal:
            self.ln_rate = self.label_noise / num_epochs

        if self.discrim_noise_linear_anneal:
            self.dn_rate = self.discrim_noise / num_epochs

        print("Beginning training")
        og_start_time = time.time()
        start_time = time.time()
        for epoch in range(num_epochs):
            for i in range(cadence):
                for x, y in self.data_gen:
                    if device_check:
                        x, y = x.to(self.device), y.to(self.device)
                    self.train_one_step(x, y)

            self.next_epoch()

            if self.epoch % print_freq == 0 or (self.epoch == num_epochs):
                print("Time: %ds" % (time.time() - start_time))
                start_time = time.time()

                self.print_progress(total_epochs)

            if eval_freq is not None:
                if self.epoch % eval_freq == 0 or (self.epoch == num_epochs):
                    self.stored_acc.append(self.test_model(stratify=self.eval_stratify))
                    print("Epoch: %d\tEvaluator Score: %.4f" % (self.epoch, np.max(self.stored_acc[-1])))

        print("Total training time: %ds" % (time.time() - og_start_time))
        print("Training complete")

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

    def test_model(self, stratify=None):
        """
        Train a model on fake data and evaluate on test data in order to evaluate network as it trains
        :param stratify: How to proportion out the labels. If None, a straight average is used.
        :return: Tuple of list of classes trained and the scores each achieved
        """
        fake_scores = []

        for size in self.test_ranges:
            genned_data, genned_labels = self.gen_fake_data(bs=size, stratify=stratify)

            if self.data_gen.dataset.le_dict is not None:
                genned_data = self.reencode(genned_data, self.data_gen.dataset.le_dict)

            score_fake_tmp = train_test_logistic_reg(x_train=genned_data, y_train=genned_labels,
                                                     x_test=self.data_gen.dataset.x_test_arr, y_test=self.data_gen.dataset.y_test_arr,
                                                     param_grid=self.eval_param_grid, cv=self.eval_folds, random_state=self.seed,
                                                     labels_list=self.labels_list, verbose=0)

            fake_scores.append(score_fake_tmp)

            torch.save(self.netG.state_dict(), self.path + "/stored_generators/Epoch_" + str(self.epoch) + "_Generator.pt")

        return fake_scores

    def gen_fake_data(self, bs, stratify=None):
        """
        Generate fake data. Calls gen_labels method below.
        :param bs: Batch size of fake data to generate
        :param stratify: How to proportion out the labels. If None, a straight average is used.
        :return: Tuple of generated data and associated labels
        """
        noise = torch.randn(bs, self.nz, device=self.device)
        fake_labels, output_labels = self.gen_labels(bs=bs, stratify=stratify)
        fake_labels = fake_labels.to(self.device)

        self.netG.eval()
        with torch.no_grad():
            fake_data = self.netG(noise, fake_labels).cpu().detach().numpy()

        return fake_data, output_labels

    def gen_labels(self, bs, stratify=None):
        """
        Generate labels for generating fake data
        :param bs: Number of desired labels
        :param stratify: How to proportion out the labels. If None, a straight average is used.
        :return: Tuple of one hot encoded labels and the labels themselves
        """
        if stratify is None:
            stratify = [1/self.nc for i in range(self.nc)]
        counts = np.round(np.dot(stratify, bs), decimals=0).astype('int')
        while np.sum(counts) != bs:
            if np.sum(counts) > bs:
                counts[random.choice(range(self.nc))] -= 1
            else:
                counts[random.choice(range(self.nc))] += 1
        output_one_hot = np.empty((0, self.nc))
        one_hot = pd.get_dummies(self.labels_list)
        output_labels = np.empty(0)
        for i in range(self.nc):
            tmp_one_hot = np.empty((counts[i], self.nc))
            tmp_labels = np.full(counts[i], self.labels_list[i])
            output_labels = np.concatenate((output_labels, tmp_labels), axis=0)
            for j in range(self.nc):
                tmp_one_hot[:, j] = one_hot.iloc[i, j]
            output_one_hot = np.concatenate((output_one_hot, tmp_one_hot), axis=0)
            output_one_hot = torch.tensor(output_one_hot, dtype=torch.float)
        return output_one_hot, output_labels

    def reencode(self, raw_fake_output, le_dict):
        """
        Reencode categorical variables with the label encoder
        :param raw_fake_output: Data generated by netG
        :param le_dict: Dictionary of LabelEncoders to be used for inverse transformation back to original raw data
        :return: Generated data inverse transformed and prepared for train_test_logistic_reg method. Data is still scaled and one hot encoded.
        """
        curr = 0
        new_fake_output = np.copy(raw_fake_output)
        for _, le in le_dict.items():
            n = len(le.classes_)
            newcurr = curr + n
            max_idx = np.argmax(raw_fake_output[:, curr:newcurr], 1)
            new_fake_output[:, curr:newcurr] = np.eye(n)[max_idx]
            curr = newcurr
        return new_fake_output

    def rev_ohe_le_scaler(self, data, genned_labels, dep_var, preprocessed_cat_mask, ohe, le_dict, scaler, cat_inputs, cont_inputs, int_inputs):
        """
        :param data: Output of reencode method
        :param genned_labels: Labels corresponding to generated data
        :return: Generated data fully inverse transformed to be on the same basis as the original raw data
        """
        df = pd.DataFrame(index=range(data.shape[0]), columns=[dep_var] + list(cat_inputs) + list(cont_inputs))

        # Add labels
        df[dep_var] = genned_labels.astype('int')

        # Split into cat and cont variables
        cat_arr = data[:, preprocessed_cat_mask]
        cont_arr = data[:, ~preprocessed_cat_mask]

        # Inverse transform categorical variables
        numerics = ohe.inverse_transform(cat_arr)
        for i, le in enumerate(le_dict.items()):
            df[le[0]] = le[1].inverse_transform(numerics[:, i].astype('int'))

        # Inverse transform continuous variables
        og_cont_arr = scaler.inverse_transform(cont_arr)
        df[cont_inputs] = og_cont_arr

        # Round integer inputs
        df[int_inputs] = df[int_inputs].round(decimals=0).astype('int')

        return df

    def print_progress(self, total_epochs):
        """Print metrics of interest"""
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (self.epoch, total_epochs, self.netD.losses[-1], self.netG.losses[-1], self.netD.Avg_D_reals[-1], self.netD.Avg_D_fakes[-1], self.netG.Avg_G_fakes[-1]))

    def next_epoch(self):
        """Run netG and netD methods to prepare for next epoch. Mostly saves histories and resets history collection objects."""
        self.epoch += 1

        self.netG.next_epoch()
        self.netG.next_epoch_gen()

        self.netD.next_epoch()
        self.netD.next_epoch_discrim()

        # Anneal noise rates
        self.label_noise -= self.ln_rate
        self.discrim_noise -= self.dn_rate
        self.netD.noise = GaussianNoise(device=self.device, sigma=self.discrim_noise)

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
        num_tests = len(self.test_ranges)

        ys = np.empty((length, num_tests))
        xs = np.empty((length, num_tests))
        barWidth = 1 / (num_tests + 1)
        for i in range(num_tests):
            ys[:, i] = np.array([x[i] for x in self.stored_acc])
            xs[:, i] = np.arange(length) + barWidth * (i + 1)
            plt.bar(xs[:, i], ys[:, i], width=barWidth, edgecolor='white', label=self.test_ranges[i])

        plt.plot(np.linspace(0, length, length), np.full(length, benchmark_acc), linestyle='dashed', color='r')

        plt.xlabel('Epoch', fontweight='bold')
        plt.xticks([r + barWidth for r in range(length)], list(range(1, length + 1)))
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

    def load_netG(self, best=False, epoch=None):
        """Load a previously stored netG"""
        assert best or epoch is not None, "Either best arg must be True or epoch arg must not be None"

        if best:
            def parse_epoch(x):
                pattern = re.compile(r"[0-9]+")
                return int(re.findall(pattern=pattern, string=x)[0])
            gens = os.listdir(os.path.join(self.path, "stored_generators"))
            gens = sorted(gens, key=parse_epoch)
            epoch = parse_epoch(gens[np.argmax(self.stored_acc) // len(self.test_ranges)])

        self.netG.load_state_dict(torch.load(self.path + "/stored_generators/Epoch_" + str(epoch) + "_Generator.pt"))

    def gen_data(self, size, stratify=None):
        """Generates a data set formatted like the original data"""
        genned_data, genned_labels = self.gen_fake_data(bs=size, stratify=stratify)
        genned_data = self.reencode(genned_data, self.data_gen.dataset.le_dict)
        genned_data_df = self.rev_ohe_le_scaler(data=genned_data,
                                                genned_labels=genned_labels,
                                                dep_var=self.data_gen.dataset.dep_var,
                                                preprocessed_cat_mask=self.data_gen.dataset.preprocessed_cat_mask,
                                                ohe=self.data_gen.dataset.ohe,
                                                le_dict=self.data_gen.dataset.le_dict,
                                                scaler=self.data_gen.dataset.scaler,
                                                cat_inputs=self.data_gen.dataset.cat_inputs,
                                                cont_inputs=self.data_gen.dataset.cont_inputs,
                                                int_inputs=self.data_gen.dataset.int_inputs)
        return genned_data_df

    def draw_architecture(self, net, show, save):
        """
        Utilizes torchviz to print current graph to a pdf
        :param net: Network to draw graph for. Either netG or netD
        :param show: Whether to show the graph. To visualize in jupyter notebooks, run the returned viz.
        :param save: Where to save the graph.
        """
        assert net in {self.netG, self.netD}, "Invalid entry for net. Should be netG or netD"

        if save is None:
            save = self.path

        iterator = iter(self.data_gen)
        x, y = next(iterator)
        x, y = x.to(self.device), y.to(self.device).type(torch.float32)

        if net == self.netG:
            noise = torch.randn(x.shape[0], self.nz, device=self.device)
            viz = make_dot(net(noise, y), params=dict(net.named_parameters()))
        else:
            viz = make_dot(net(x, y), params=dict(net.named_parameters()))

        if net == self.netG:
            title = "Generator"
        else:
            title = "Discriminator"

        safe_mkdir(save + "/architectures")
        viz.render(filename=save + "/architectures/" + title, view=show)

        return viz
