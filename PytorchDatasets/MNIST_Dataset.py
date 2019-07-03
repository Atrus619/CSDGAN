from torch.utils import data


class MNIST_Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# TODO: Use a generator instead of a loader
class Fake_MNIST_Dataset(data.Dataset):
    def __init__(self, netG, labels, size):
        self.netG = netG
        self.labels = labels
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        pass