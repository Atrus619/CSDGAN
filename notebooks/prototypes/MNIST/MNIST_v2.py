from CSDGAN.classes.image.ImageCGAN import ImageCGAN
import configs.MNIST as cfg
from utils.data_loading import *
from CSDGAN.classes.image.ImageDataset import ImageDataset, GeneratedImageDataset
from torch.utils import data
import os
import pickle
import random
import numpy as np
from utils.image_utils import img_dataset_preprocessor, show_real_grid, augment


# Set random seem for reproducibility
print("Random Seed: ", cfg.MANUAL_SEED)
random.seed(cfg.MANUAL_SEED)
torch.manual_seed(cfg.MANUAL_SEED)

# Ensure directory exists for outputs
exp_path = os.path.join("experiments", cfg.EXPERIMENT_NAME)

# Import data and split
mnist = load_processed_dataset('MNIST')
x_comb, y_comb = torch.cat((mnist[0][0], mnist[1][0]), 0).numpy(), torch.cat((mnist[0][1], mnist[1][1]), 0).numpy()
x_train, y_train, x_val, y_val, x_test, y_test, le, ohe = img_dataset_preprocessor(x=x_comb, y=y_comb, splits=cfg.SPLITS, seed=cfg.MANUAL_SEED)

# Automatically determine these parameters
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")  # GPU if exists, else CPU
x_dim = (x_train.shape[1], x_train.shape[2])  # Dimensions of input images

# Print resulting sizes
print("Train Set Size:", x_train.shape[0])
print("Validation Set Size:", x_val.shape[0])
print("Test Set Size:", x_test.shape[0])
print("Each image size:", x_dim[0], "x", x_dim[1])

# Define generators
train_dataset = ImageDataset(x=x_train, y=y_train)
train_gen = data.DataLoader(train_dataset, **cfg.TRAINING_PARAMS)

val_dataset = ImageDataset(x=x_val, y=y_val)
val_gen = data.DataLoader(val_dataset, **cfg.TRAINING_PARAMS)

test_dataset = ImageDataset(x=x_test, y=y_test)
test_gen = data.DataLoader(test_dataset, **cfg.TRAINING_PARAMS)

# Define GAN
CGAN = ImageCGAN(train_gen=train_gen,
                 val_gen=val_gen,
                 test_gen=test_gen,
                 le=le,
                 ohe=ohe,
                 device=device,
                 path=exp_path,
                 **cfg.CGAN_INIT_PARAMS)

# Check performance on real data
try:
    benchmark_acc, real_netE = CGAN.eval_on_real_data(num_epochs=cfg.CGAN_INIT_PARAMS['eval_num_epochs'], es=cfg.CGAN_INIT_PARAMS['early_stopping_patience'])
except RuntimeError:
    benchmark_acc, real_netE = CGAN.eval_on_real_data(num_epochs=cfg.CGAN_INIT_PARAMS['eval_num_epochs'], es=cfg.CGAN_INIT_PARAMS['early_stopping_patience'])
print("Benchmark Accuracy:", benchmark_acc)

# Train CGAN
try:
    CGAN.train_gan(num_epochs=cfg.NUM_EPOCHS, print_freq=cfg.PRINT_FREQ, eval_freq=cfg.EVAL_FREQ)
except RuntimeError:
    CGAN.train_gan(num_epochs=cfg.NUM_EPOCHS, print_freq=cfg.PRINT_FREQ, eval_freq=cfg.EVAL_FREQ)

# Display final grid
show_real_grid(x_train=x_train, y_train=np.argmax(y_train, 1), nc=10, num_channels=1, grid_rows=10, x_dim=(28, 28))

# Generate sample images
CGAN.show_img(0)

# Diagnostics
CGAN.run_all_diagnostics(real_netE=real_netE, benchmark_acc=benchmark_acc, save=exp_path, show=True)

# Save model
with open(exp_path + "/CGAN.pkl", 'wb') as f:
    pickle.dump(CGAN, f)

# Load model
with open(exp_path + "/CGAN.pkl", 'rb') as f:
    CGAN = pickle.load(f)

# Test Grad CAM
num = 9

CGAN.draw_cam(gen=CGAN.data_gen, net=CGAN.netE, label=num, mistake=False, path=exp_path + "/plswork.jpg", show=True)
CGAN.draw_cam(gen=CGAN.data_gen, net=CGAN.netE, label=num, mistake=True, path=exp_path + "/plswork.jpg", show=True)

CGAN.draw_cam(gen=CGAN.data_gen, net=CGAN.netD, label=3, mistake=True, path=exp_path + "/plswork.jpg", show=True)
CGAN.draw_cam(gen=CGAN.netG, net=CGAN.netE, label=3, mistake=False, path=exp_path + "/plswork.jpg", show=True)
CGAN.draw_cam(gen=CGAN.netG, net=CGAN.netE, label=3, mistake=True, path=exp_path + "/plswork.jpg", show=True)

# Test drawing architectures
CGAN.draw_architecture(net=CGAN.netG, show=True, save="test")
CGAN.draw_architecture(net=CGAN.netD, show=True, save="test")
CGAN.draw_architecture(net=CGAN.netE, show=True, save="test")

# Data augmentation
augment(train_dataset, 10000)

generator_augmented_training_set = GeneratedImageDataset(CGAN.netG, CGAN.fake_data_set_size, CGAN.nz, CGAN.nc, CGAN.num_channels, CGAN.fake_bs, CGAN.ohe, CGAN.device, CGAN.x_dim)
augment(generator_augmented_training_set, 10000)
