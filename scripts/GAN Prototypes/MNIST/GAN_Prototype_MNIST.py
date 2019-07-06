from models.MNIST.CGAN import CGAN
import config.MNIST as cfg
from utils.utils import *
from utils.data_loading import *
from utils.MNIST import *
from PytorchDatasets.MNIST_Dataset import MNIST_Dataset
from torch.utils import data

# Set random seem for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Ensure directory exists for outputs
safe_mkdir(cfg.netE_FILEPATH)

# Import data and split
mnist = load_dataset('mnist')
x_comb, y_comb = torch.cat((mnist[0].data, mnist[1].data), 0).numpy(), torch.cat((mnist[0].targets, mnist[1].targets), 0).numpy()
x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x_comb, y_comb, cfg.SPLITS, manualSeed)

# Automatically determine these parameters
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")  # GPU if exists, else CPU
x_dim = (x_train.shape[1], x_train.shape[2]),  # Dimensions of input images

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
            netE_filepath=cfg.netE_FILEPATH,
            **cfg.CGAN_PARAMS)

# Check performance on real data
# try:
#     benchmark_acc = eval_on_real_data(CGAN=CGAN, num_epochs=cfg.CGAN_PARAMS['eval_num_epochs'], es=cfg.CGAN_PARAMS['early_stopping_patience'])
# except RuntimeError:
#     benchmark_acc = eval_on_real_data(CGAN=CGAN, num_epochs=cfg.CGAN_PARAMS['eval_num_epochs'], es=cfg.CGAN_PARAMS['early_stopping_patience'])
# print(benchmark_acc)

# Train CGAN
try:
    CGAN.train_gan(30, 5)
except RuntimeError:
    CGAN.train_gan(30, 5)

# Display final grid
CGAN.show_grid(-1)

# Generate sample images
CGAN.show_img(0)

# Display video of progress
CGAN.show_video()

# TODO: Better visualization of training (add labels to grid, visualize progress, visualize weights and such)
# TODO: Seems to perform poorly at generating conditional numbers
# TODO: Add comments to each function
# TODO: Add augmentation and compare
# TODO: Possibly bad initialization too
# TODO: Add time tracking to training loop for CGAN
