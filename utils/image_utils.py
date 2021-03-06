import utils.utils as uu

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torchvision.transforms as t


def img_dataset_preprocessor(x, y, splits, seed=None):
    """Min-max scales an image data set so that all values lie between 0 and 1. Also splits data into train/val/test partitions."""
    y, le, ohe = uu.encode_y(y)

    x = x.astype('float32')

    x_train, y_train, x_val, y_val, x_test, y_test = uu.train_val_test_split(x, y, splits=splits, random_state=seed)

    train_max = x_train.max()
    train_min = x_train.min()

    def min_max_scaler(tensor, max, min):
        return (tensor - min) / (max - min)

    x_train = min_max_scaler(tensor=x_train, max=train_max, min=train_min)
    x_val = min_max_scaler(tensor=x_val, max=train_max, min=train_min)
    x_test = min_max_scaler(tensor=x_test, max=train_max, min=train_min)

    return x_train, y_train, x_val, y_val, x_test, y_test, le, ohe


def convert_y_to_one_hot(y, nc):
    """Converts a tensor of labels to a one_hot encoded version"""
    new_y = torch.zeros([len(y), nc], dtype=torch.uint8, device='cpu')
    y = y.view(-1, 1)
    new_y.scatter_(1, y, 1)
    return new_y


def show_real_grid(x_train, y_train, nc, num_channels, grid_rows, x_dim):
    """Generates a grid of images on real data. Randomly selects examples from provided args."""
    pre_grid = torch.empty((nc*grid_rows, num_channels, x_dim[0], x_dim[1]))
    for i in range(nc):
        # Collect grid_rows random samples of each label
        locs = np.random.choice(np.where(y_train == i)[0], grid_rows, replace=False)
        pre_grid[nc*i:(nc*(i+1)), 0, :] = x_train[locs]

    grid = vutils.make_grid(tensor=pre_grid, nrow=grid_rows, normalize=True)

    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()


def pow_2(n, first):
    return np.floor(np.log(n / first)/np.log(2))


def find_pow_2_arch(dim):
    """
    For a specified dimension length of an image (number of pixels length or width),
    calculates the minimum number of pixels to crop in order to construct a reasonable neural network architecture.
    """
    firsts = {3, 5, 7, 9, 11}
    # Initialize to something basic
    best_crop = 999
    best_first = 3
    for first in firsts:
        crop = dim - pow(2, pow_2(dim, first)) * first
        if crop < best_crop:
            best_crop = crop
            best_first = first
    return int(best_crop), int(best_first), int(pow_2(dim, best_first))


# Block methods below define the layers needed to compose the required architecture based on the amount of upsampling
def first_block(h, w, in_channels, out_channels):
    """First block, sets dimensions to be an odd number less than 11"""
    viable_options = {3, 5, 7, 9, 11}
    assert h in viable_options and w in viable_options, "Please make sure w and h are viable options"

    ct2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(w, h), stride=1, padding=0, output_padding=0, bias=True)
    ct2_bn = nn.BatchNorm2d(out_channels)

    return ct2, ct2_bn


def ct2_upsample_block(h, w, in_channels, out_channels, add_op=(0, 0)):
    """2x upsample if h=2 or w=2, 1x upsample if equal to 1. Final block will get additional output_padding (add_op)"""
    assert h in {1, 2} and w in {1, 2}, "Invalid value for h or w"

    output_padding = h - 1 + add_op[0], w - 1 + add_op[1]

    ct2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=(h, w), padding=2, output_padding=output_padding, bias=True)
    ct2_bn = nn.BatchNorm2d(out_channels)

    return ct2, ct2_bn


def cn2_downsample_block(h, w, in_channels, out_channels):
    """2x downsample if h=2 or w=2, 1x downsample if equal to 1."""
    assert h in {1, 2} and w in {1, 2}, "Invalid value for h or w"

    kernel_size = (h + 2, w + 2)
    cn2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=(h, w), padding=1, bias=True)
    cn2_bn = nn.BatchNorm2d(out_channels)

    return cn2, cn2_bn


def update_h_w_curr(h_rem, w_rem):
    """Helper function to downsample h and w by 2 if currently greater than 2"""
    h_curr, w_curr = min(h_rem, 2), min(w_rem, 2)
    h_rem = h_rem // 2 if h_rem > 1 else 1
    w_rem = w_rem // 2 if w_rem > 1 else 1

    return h_rem, w_rem, h_curr, w_curr


def evaluator_cn2_block(h, w, in_channels, out_channels):
    """Incorporates max pooling instead of strided downsampling"""
    assert h in {1, 2} and w in {1, 2}, "Invalid value for h or w"

    cn2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
    cn2_bn = nn.BatchNorm2d(out_channels)
    cn2_mp = nn.MaxPool2d(kernel_size=(h, w), stride=(h, w))

    return cn2, cn2_bn, cn2_mp


def augment(dataset, n):
    """
    Increases the data set by augmenting a stratified sample of size n through the transfm_batch method.
    Intended to be used with the MNIST data set.
    Modifies data set in place.
    """
    x_aug, _, y_aug, _, y_aug_raw, _ = train_test_split(dataset.x.numpy(), dataset.y.numpy(), np.argmax(dataset.y.numpy(), 1),
                                                        test_size=dataset.x.shape[0]-n, stratify=np.argmax(dataset.y.numpy(), 1))

    x_aug, y_aug = torch.from_numpy(x_aug), torch.from_numpy(y_aug)
    x_aug = trnsfm_batch(x_aug, y_aug_raw)
    dataset.x, dataset.y = torch.cat((dataset.x, x_aug), dim=0), torch.cat((dataset.y, y_aug), dim=0)


def trnsfm_batch(img, labels):
    """Performs various transformations in order to augment the data set. Intended to be used with the MNIST data set."""
    PIL = t.ToPILImage()
    TNSR = t.ToTensor()
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
