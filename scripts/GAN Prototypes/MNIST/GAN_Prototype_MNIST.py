from sklearn.model_selection import train_test_split
from models.CGAN_mnist import CGAN_Generator, CGAN_Discriminator
from scripts.Utils.utils import *
from scripts.Utils.data_loading import *
from PytorchDatasets.MNIST_Dataset import MNIST_Dataset
from torch.utils import data
import matplotlib.pyplot as plt
import random

# Set random seem for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Import data
mnist = load_dataset('mnist')
x_train, y_train, x_test, y_test = mnist[0].data, mnist[0].targets, mnist[1].data, mnist[1].targets
x_train, x_val, y_train, y_val = train_test_split(x_train.numpy(), y_train.numpy(), test_size=0.2, stratify=y_train.numpy(), random_state=manualSeed)
x_train, y_train, x_val, y_val = torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(x_val), torch.from_numpy(y_val)
print(y_train[0])
plt.imshow(x_train[0], cmap='gray')

# Define generators
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

training_set = MNIST_Dataset(x_train, y_train)
training_generator = data.DataLoader(training_set, **params)

validation_set = MNIST_Dataset(x_val, y_val)
validation_generator = data.DataLoader(validation_set, **params)
