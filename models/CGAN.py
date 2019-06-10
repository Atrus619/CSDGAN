import torch.nn as nn
import torch


# Generator class
class CGAN_Generator(nn.Module):
    def __init__(self, nz, H, out_dim, nc):
        super(CGAN_Generator, self).__init__()
        self.fc1_1 = nn.Linear(nz, H, bias=True)
        # self.fc1_1_bn = nn.BatchNorm1d(H)
        self.fc1_2 = nn.Linear(nc, H, bias=True)
        # self.fc1_2_bn = nn.BatchNorm1d(H)
        self.output = nn.Linear(2*H, out_dim, bias=True)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, noise, labels):
        """
        Single dense hidden layer network.
        TODO: May want to adjust ReLU or add batch normalization later.
        :param noise: Random Noise vector Z
        :param labels: Label embedding
        :return: Row of data for iris data set (4 real values)
        """
        x = self.act(self.fc1_1(noise))
        y = self.act(self.fc1_2(labels))
        x = torch.cat([x, y], -1)
        return self.output(x)  # TODO: Make sure it is appropriate to not use an activation here


# Discriminator class
class CGAN_Discriminator(nn.Module):
    def __init__(self, H, out_dim, nc):
        super(CGAN_Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(out_dim, H, bias=True)
        # self.fc1_1_bn = nn.BatchNorm1d(H)
        self.fc1_2 = nn.Linear(nc, H, bias=True)
        # self.fc1_2_bn = nn.BatchNorm1d(H)
        self.output = nn.Linear(2*H, 1, bias=True)
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Sigmoid()

    def forward(self, row, labels):
        """
        Single dense hidden layer network.
        TODO: May want to adjust ReLU or add batch normalization later.
        :param row: Row of data from iris data set (4 real values)
        :param labels: Label embedding
        :return: Binary classification (sigmoid activation on a single unit hidden layer)
        """
        x = self.act(self.fc1_1(row))
        y = self.act(self.fc1_2(labels))
        x = torch.cat([x, y], -1)
        return self.m(self.output(x))
