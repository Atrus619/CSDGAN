from torch.utils import data
from utils.MNIST import *
import torchvision
import torchvision.transforms as t
from sklearn.model_selection import train_test_split


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


class Augmented_MNIST_Dataset(data.Dataset):
    def __init__(self, x, y, n):
        assert n < x.shape[0], "Amount of augmentation n must be less than total data set size"
        self.x = x.view(-1, 1, 28, 28).float() / 255
        self.y_raw = y
        self.y = convert_y_to_one_hot(y)
        self.augment(n)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def augment(self, n):
        """Increases the data set by augmenting a stratified sample of size n through the transfm_batch method"""
        x_aug, _, y_aug, _, y_aug_raw, _ = train_test_split(self.x.numpy(), self.y.numpy(), self.y_raw.numpy(), test_size=self.x.shape[0]-n, stratify=self.y_raw)
        x_aug, y_aug = torch.from_numpy(x_aug), torch.from_numpy(y_aug)
        x_aug = self.trnsfm_batch(x_aug, y_aug_raw)
        self.x, self.y = torch.cat((self.x, x_aug), dim=0), torch.cat((self.y, y_aug), dim=0)

    @staticmethod
    def trnsfm_batch(img, labels):
        """Performs various transformations in order to augment the data set"""
        PIL = torchvision.transforms.ToPILImage()
        TNSR = torchvision.transforms.ToTensor()
        crop_trnsfm = t.RandomResizedCrop(28, scale=(0.75, 1.0), ratio=(0.75, 1.3333))
        affine_trnsfm = t.RandomAffine((-15, 15))
        vert_trnsfm = t.RandomVerticalFlip(p=0.5)
        hor_trnsfm = t.RandomHorizontalFlip(p=0.5)
        final_trnsfm = t.Compose([crop_trnsfm, affine_trnsfm])
        spcl_trnsfm = t.Compose([vert_trnsfm, hor_trnsfm])
        spcl_list = [1, 8]
        out = torch.empty_like(img)
        for i in range(img.shape[0]):
            tmp = img[i].view(28, 28)
            tmp = PIL(tmp)
            tmp = final_trnsfm(tmp)
            if labels[i] in spcl_list:
                tmp = spcl_trnsfm(tmp)
            tmp = TNSR(tmp)
            out[i] = tmp
        return out


class Generator_Augmented_MNIST_Dataset(data.Dataset):
    def __init__(self, x, y, n, netG):
        assert n % 100 == 0, "Please make sure n is divisible by 100 so classes can be perfectly balanced"

        self.x = x.view(-1, 1, 28, 28).float() / 255
        self.y_raw = y
        self.y = convert_y_to_one_hot(y)
        self.netG = netG
        self.augment(n)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def augment(self, n):
        """Increases the data set by generating a stratified sample of size n using netG"""
        bs = self.netG.fixed_labels.shape[0]

        noise = torch.randn(bs, self.netG.nz, device=self.netG.device)
        x_aug = torch.empty((n, self.netG.num_channels, self.netG.x_dim[0], self.netG.x_dim[1]), dtype=torch.float32, device=self.netG.device)
        y_aug = torch.empty((n, self.netG.fixed_labels.shape[1]), dtype=torch.float32, device=self.netG.device)

        self.netG.eval()
        for i in range(n // bs):
            with torch.no_grad():
                x_aug[(100*i):(100*(i+1))] = self.netG(noise, self.netG.fixed_labels)
            y_aug[(100 * i):(100 * (i + 1))] = self.netG.fixed_labels.detach()

        self.x, self.y = torch.cat((self.x, x_aug.cpu()), dim=0), torch.cat((self.y, y_aug.type(torch.uint8).cpu()), dim=0)
