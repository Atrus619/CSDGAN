from models.MNIST.CGAN import CGAN
import config.MNIST as cfg
from utils.utils import *
from utils.data_loading import *
from utils.MNIST import *
from PytorchDatasets.MNIST_Dataset import MNIST_Dataset
from torch.utils import data
import os
import pickle

# Set random seem for reproducibility
print("Random Seed: ", cfg.MANUAL_SEED)
random.seed(cfg.MANUAL_SEED)
torch.manual_seed(cfg.MANUAL_SEED)

# Ensure directory exists for outputs
exp_path = os.path.join("experiments", cfg.EXPERIMENT_NAME)
eval_path = os.path.join(exp_path, "stored_evaluators")
safe_mkdir(exp_path)
safe_mkdir(eval_path)

# Import data and split
mnist = load_dataset('mnist')
x_comb, y_comb = torch.cat((mnist[0].data, mnist[1].data), 0).numpy(), torch.cat((mnist[0].targets, mnist[1].targets), 0).numpy()
x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x_comb, y_comb, cfg.SPLITS, cfg.MANUAL_SEED)

# Automatically determine these parameters
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")  # GPU if exists, else CPU
x_dim = (x_train.shape[1], x_train.shape[2])  # Dimensions of input images

# Print resulting sizes
print("Train:", x_train.shape)
print("Validate: ", x_val.shape)
print("Test: ", x_test.shape)

# Print an example image
print(y_train[0])
# plt.imshow(x_train[0], cmap='gray')

# Define generators
training_set = MNIST_Dataset(x_train, y_train)
training_generator = data.DataLoader(training_set, **cfg.TRAINING_PARAMS)

validation_set = MNIST_Dataset(x_val, y_val)
validation_generator = data.DataLoader(validation_set, **cfg.TRAINING_PARAMS)

test_set = MNIST_Dataset(x_test, y_test)
test_generator = data.DataLoader(test_set, **cfg.TRAINING_PARAMS)

# Define GAN
CGAN = CGAN(train_gen=training_generator,
            val_gen=validation_generator,
            test_gen=test_generator,
            device=device,
            x_dim=x_dim,
            netE_filepath=eval_path,
            **cfg.CGAN_INIT_PARAMS)

# Check performance on real data
try:
    benchmark_acc, real_netE = eval_on_real_data(CGAN=CGAN, num_epochs=cfg.CGAN_INIT_PARAMS['eval_num_epochs'], es=cfg.CGAN_INIT_PARAMS['early_stopping_patience'])
except RuntimeError:
    benchmark_acc, real_netE = eval_on_real_data(CGAN=CGAN, num_epochs=cfg.CGAN_INIT_PARAMS['eval_num_epochs'], es=cfg.CGAN_INIT_PARAMS['early_stopping_patience'])
benchmark_acc = benchmark_acc.numpy().take(0)
print(benchmark_acc)

# Train CGAN
try:
    CGAN.train_gan(num_epochs=cfg.NUM_EPOCHS, print_freq=cfg.PRINT_FREQ, eval_freq=cfg.EVAL_FREQ)
except RuntimeError:
    CGAN.train_gan(num_epochs=cfg.NUM_EPOCHS, print_freq=cfg.PRINT_FREQ, eval_freq=cfg.EVAL_FREQ)

# Display final grid
CGAN.show_grid(-1)
show_real_grid(x_train, y_train)

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

# TODO: Add augmentation and compare
# TODO: Use FAR LESS training data (only a handful of examples)
# TODO: Create notebook
