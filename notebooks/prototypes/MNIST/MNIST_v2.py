from classes.Image.ImageCGAN import ImageCGAN
import configs.MNIST as cfg
from utils.utils import *
from utils.data_loading import *
from utils.MNIST import *
from classes.Image.ImageDataset import ImageDataset, GeneratedImageDataset
from torch.utils import data
import os
import pickle
import random
from utils.ImageUtils import img_dataset_preprocesser


# Set random seem for reproducibility
print("Random Seed: ", cfg.MANUAL_SEED)
random.seed(cfg.MANUAL_SEED)
torch.manual_seed(cfg.MANUAL_SEED)

# Ensure directory exists for outputs
exp_path = os.path.join("experiments", cfg.EXPERIMENT_NAME)

# Import data and split
mnist = load_processed_dataset('MNIST')
x_comb, y_comb = torch.cat((mnist[0][0], mnist[1][0]), 0).numpy(), torch.cat((mnist[0][1], mnist[1][1]), 0).numpy()
x_train, y_train, x_val, y_val, x_test, y_test, le, ohe = img_dataset_preprocesser(x=x_comb, y=y_comb, splits=cfg.SPLITS, seed=cfg.MANUAL_SEED)

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
                 x_dim=x_dim,
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
# CGAN.show_grid(-1)
# show_real_grid(x_train, y_train)

# Generate sample images
# CGAN.show_img(0)

# Diagnostics
CGAN.run_all_diagnostics(real_netE=real_netE, benchmark_acc=benchmark_acc, save=exp_path, show=True)

# Save model
with open(exp_path + "/CGAN.pkl", 'wb') as f:
    pickle.dump(CGAN, f)

# Load model
with open(exp_path + "/CGAN.pkl", 'rb') as f:
    CGAN = pickle.load(f)

# Test Grad CAM
x, y = CGAN.test_gen.__iter__().__next__()
CGAN.netD.draw_cam(img=x[0], label=y[0], path=exp_path + "/plswork.jpg")

CGAN.init_evaluator(CGAN.train_gen, CGAN.val_gen)
CGAN.netE.draw_cam(img=x[1], path=exp_path + "/plswork2.jpg")

x = CGAN.find_particular_img(CGAN.train_gen, "D", 3, True)
CGAN.netD.draw_cam(img=x, label=3, path=exp_path + "/plswork.jpg")

CGAN.draw_cam(gen=CGAN.train_gen, net="E", label=3, mistake=True, path=exp_path + "/plswork.jpg", show=True)

# Test drawing architectures
CGAN.draw_architecture(net=CGAN.netG, show=True, save="test")
CGAN.draw_architecture(net=CGAN.netD, show=True, save="test")
CGAN.draw_architecture(net=CGAN.netE, show=True, save="test")

