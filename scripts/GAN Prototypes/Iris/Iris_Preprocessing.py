from scripts.Utils.data_loading import load_dataset
from sklearn.model_selection import train_test_split
from scripts.Utils.utils import *
import random

# Set random seem for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

iris = load_dataset('iris')
iris.head()

# Split 50-50 so we can demonstrate the effectiveness of additional data
x_train, x_test, y_train, y_test = train_test_split(iris.drop(columns='species'), iris.species, test_size=0.5, stratify=iris.species, random_state=manualSeed)

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

# Scale inputs
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train_tensor = torch.tensor(x_train, dtype=torch.float)
y_train_dummies = pd.get_dummies(y_train)
y_train_dummies_tensor = torch.tensor(y_train_dummies.values, dtype=torch.float)

# if __name__ == "__main__":
