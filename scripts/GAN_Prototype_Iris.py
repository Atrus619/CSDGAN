from scripts.data_loading import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.VGAN import VGAN_Generator, VGAN_Discriminator
from models.CGAN import CGAN_Generator, CGAN_Discriminator
from utils import *
import random

# TODO: Add noise to discriminator inputs
# TODO: Wasserstein
# TODO: Feature matching
# TODO: Debug to make sure CGAN is working as intended

# TODO: Convert labels to torch.nn.embedding? https://torchgan.readthedocs.io/en/stable/_modules/torchgan/models/conditional.html#ConditionalGANGenerator
# TODO: Possibly change how inputs are concatenated? I think I should add them at the start!

# Set random seem for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

iris = load_dataset('iris')
iris.head()

# Split 50-50 so we can demonstrate the effectiveness of additional data
x_train, x_test, y_train, y_test = train_test_split(iris.drop(columns='species'), iris.species, test_size=0.5, stratify=iris.species, random_state=42)

# Parameters
nz = 32  # Size of generator noise input  # TODO: May need to mess around with this later
H = 16  # Size of hidden network layer
out_dim = x_train.shape[1]  # Size of output
bs = x_train.shape[0]  # Full data set
nc = 3  # 3 different types of label in this problem
num_batches = 1
num_epochs = 10000

# Adam optimizer hyperparameters
lr = 2e-4
beta1 = 0.5
beta2 = 0.999

# Set the device
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Normalize inputs
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train_tensor = torch.tensor(x_train, dtype=torch.float)
y_train_dummies = pd.get_dummies(y_train)
y_train_dummies_tensor = torch.tensor(y_train_dummies.values, dtype=torch.float)

# Instantiate nets
netG = CGAN_Generator(nz=nz, H=H, out_dim=out_dim, nc=nc, bs=bs, lr=lr, beta1=beta1, beta2=beta2).to(device)
netD = CGAN_Discriminator(H=H, out_dim=out_dim, nc=nc, lr=lr, beta1=beta1, beta2=beta2).to(device)

# Print models
print(netG)
print(netD)

# Define labels
real_label = 1
fake_label = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i in range(num_batches):  # Only one batch per epoch since our data is horrifically small
        # Update Discriminator
        # All real batch first
        real_data = x_train_tensor.to(device)  # Format batch (entire data set in this case)
        real_classes = y_train_dummies_tensor.to(device)
        label = torch.full((bs,), real_label, device=device)  # All real labels

        output = netD(real_data, real_classes).view(-1)  # Forward pass with real data through Discriminator
        netD.train_one_step_real(output, label)

        # All fake batch next
        noise = torch.randn(bs, nz, device=device)  # Generate batch of latent vectors
        fake = netG(noise, real_classes)  # Fake image batch with netG
        label.fill_(fake_label)
        output = netD(fake.detach(), real_classes).view(-1)
        netD.train_one_step_fake(output, label)
        netD.combine_and_update_opt()
        netD.update_history()

        # Update Generator
        label.fill_(real_label)  # Reverse labels, fakes are real for generator cost
        output = netD(fake, real_classes).view(-1)  # Since D has been updated, perform another forward pass of all-fakes through D
        netG.train_one_step(output, label)
        netG.update_history()

        # Output training stats
        if epoch % 1000 == 0 or (epoch == num_epochs-1):
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+1, num_epochs, netD.loss.item(), netG.loss.item(), netD.D_x, netD.D_G_z1, netG.D_G_z2))
            with torch.no_grad():
                fake = netG(netG.fixed_noise, real_classes).detach().cpu()
            netG.fixed_noise_outputs.append(scaler.inverse_transform(fake))

# Output plots
training_plots(netD=netD, netG=netG, num_epochs=num_epochs)

# Train various models with real/fake data.
y_test_dummies = pd.get_dummies(y_test)
print("Dummy columns match?", all(y_train_dummies.columns == y_test_dummies.columns))
x_test = scaler.transform(x_test)
labels_list = [x for x in y_train_dummies.columns]
param_grid = {'tol': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
              'C': [0.5, 0.75, 1, 1.25],
              'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}

model_real, score_real = train_test_logistic_reg(x_train, y_train, x_test, y_test, param_grid=param_grid, cv=5, random_state=manualSeed, labels=labels_list)

# Generate various levels of amounts of fake data and test how training compares
test_range = [150]
fake_bs = 75
fake_models = []
fake_scores = []
for size in test_range:
    num_batches = size // fake_bs + 1
    genned_data = np.empty((0, out_dim))
    genned_labels = np.empty(0)
    rem = size
    while rem > 0:
        curr_size = min(fake_bs, rem)
        noise = torch.randn(curr_size, nz, device=device)
        fake_labels, output_labels = gen_labels(size=curr_size, num_classes=nc, labels_list=labels_list)
        fake_labels = fake_labels.to(device)
        rem -= curr_size
        fake_data = netG(noise, fake_labels).cpu().detach().numpy()
        genned_data = np.concatenate((genned_data, fake_data))
        genned_labels = np.concatenate((genned_labels, output_labels))
    print("For size of:", size)
    model_fake_tmp, score_fake_tmp = train_test_logistic_reg(genned_data, genned_labels, x_test, y_test,
                                                             param_grid=param_grid, cv=5, random_state=manualSeed, labels=labels_list)
    fake_models.append(model_fake_tmp)
    fake_scores.append(score_fake_tmp)

# Visualize distributions
plot_scatters(genned_data, genned_labels, "Fake Data", scaler)  # Fake data
plot_scatters(iris.drop(columns='species'), np.array(iris.species), "Full Real Data Set")  # All real data
plot_scatters(x_train, np.array(y_train), "Training Data", scaler)  # Real train data
plot_scatters(x_test, np.array(y_test), "Testing Data", scaler)  # Real test data

plot_densities(genned_data, genned_labels, "Fake Data", scaler)  # Fake data
plot_densities(iris.drop(columns='species'), np.array(iris.species), "Full Real Data Set")  # All real data
plot_densities(x_train, np.array(y_train), "Training Data", scaler)  # Real train data
plot_densities(x_test, np.array(y_test), "Testing Data", scaler)  # Real test data
