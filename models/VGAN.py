import torch.nn as nn
# TODO: Make this similar to CGAN (after confirming it works)


# Generator class
class VGAN_Generator(nn.Module):
    def __init__(self, nz, H, out_dim):
        super(VGAN_Generator, self).__init__()
        self.hidden1 = nn.Linear(nz, H, bias=True)
        self.output = nn.Linear(H, out_dim, bias=True)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Single dense hidden layer network.
        TODO: May want to adjust ReLU or add batch normalization later.
        :param x: Random Noise vector Z
        :return: Row of data for iris data set (4 real values)
        """
        x = self.hidden1(x)
        x = self.act(x)
        return self.output(x)  # TODO: Make sure it is appropriate to not use an activation here


# Discriminator class
class VGAN_Discriminator(nn.Module):
    def __init__(self, out_dim, H):
        super(VGAN_Discriminator, self).__init__()
        self.hidden1 = nn.Linear(out_dim, H, bias=True)
        self.output = nn.Linear(H, 1, bias=True)
        self.act = nn.LeakyReLU(0.2)
        self.m = nn.Sigmoid()

    def forward(self, x):
        """
        Single dense hidden layer network.
        TODO: May want to adjust ReLU or add batch normalization later.
        :param x: Row of data from iris data set (4 real values)
        :return: Binary classification (sigmoid activation on a single unit hidden layer)
        """
        x = self.hidden1(x)
        x = self.act(x)
        x = self.output(x)
        return self.m(x)
