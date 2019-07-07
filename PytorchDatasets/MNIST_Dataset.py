from torch.utils import data
from utils.MNIST import *


class MNIST_Dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x.view(-1, 1, 28, 28).float() / 255
        self.y = convert_y_to_one_hot(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class Fake_MNIST_Dataset(data.Dataset):
    def __init__(self, netG, size, nz, nc, device):
        assert size % 10 == 0, "Please adjust size so that it is divisible by the number of classes so that generated classes are perfectly balanced."
        self.netG = netG
        self.netG.eval()
        self.size = size
        self.nz = nz
        self.nc = nc
        self.device = device
        self.gen_x_bs = 500

        self.y = None
        self.gen_labels()

        self.x = None
        self.gen_data()

        self.x = self.x.to('cpu')
        self.y = self.y.long().to('cpu')

    def gen_labels(self):
        """Generate labels for fake training data for netE"""
        npc = self.size // self.nc  # Number of labels Per Class (npc)
        tmp = torch.empty(self.size, dtype=torch.int64)
        for i in range(self.nc):
            tmp[i*npc:(i+1)*npc] = i
        self.y = convert_y_to_one_hot(tmp).float().to(self.device)

    def gen_data(self):
        """Generate fake training data examples for netE. Requires prior run of gen_labels"""
        assert self.size % self.gen_x_bs == 0, "Please adjust size so that it is divisible by self.gen_x_bs so that an edge case isn't required."
        self.x = torch.empty((self.size, 1, 28, 28), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for i in range(self.size // self.gen_x_bs):
                noise = torch.randn(self.gen_x_bs, self.nz, device=self.device)
                self.x[i*self.gen_x_bs:(i+1)*self.gen_x_bs, ] = self.netG(noise, self.y[i*self.gen_x_bs:(i+1)*self.gen_x_bs])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index], self.y[index]


