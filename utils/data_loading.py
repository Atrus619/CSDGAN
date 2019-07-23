import kaggle
import torchvision
from classes.Image.ImageDataset import ImageFolderWithPaths
import torchvision.transforms as t
import os
import pandas as pd
import zipfile
from utils.utils import safe_mkdir, safe_dl
from utils.ImageUtils import *
import pickle as pkl
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import shutil

VALID_NAMES = {'iris', 'wine', 'titanic', 'lanl', 'MNIST', 'FashionMNIST'}


def prep_path(path):
    """Helper function to utilize safe_mkdir to safely create necessary directories for downloading data"""
    safe_mkdir(path)
    safe_mkdir(path + '/processed')
    safe_mkdir(path + '/raw')


def load_raw_dataset(name):
    """
    Check to see if data has been installed to the downloads/raw folder, and install if not.
    Load into memory the desired data set.
    For the LANL data set, it is required to have kaggle authentication set up on your computer.
    :param name: name of data set requested
    :return: data set requested (comes in various forms based on the desired data)
    """
    assert name in VALID_NAMES, 'Invalid data set requested. Please make sure name is one of ' + ', '.join(VALID_NAMES) + '.'

    safe_mkdir('downloads')
    kaggle.api.authenticate()
    path = os.path.join('downloads', name)
    path_raw = os.path.join(path, 'raw')

    if name == 'iris':
        prep_path(path)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', path_raw)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names', path_raw)
        return pd.read_csv(os.path.join(path_raw, 'iris.data'), names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'species'])

    elif name == 'wine':
        prep_path(path)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', path_raw)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names', path_raw)
        return pd.read_csv(os.path.join(path_raw, 'wine.data'), names=['class',
                                                                       'alcohol',
                                                                       'malic_acid',
                                                                       'ash',
                                                                       'alkalinity',
                                                                       'magnesium',
                                                                       'phenols',
                                                                       'flavanoids',
                                                                       'nonflavanoid_phenols',
                                                                       'proanthocyanins',
                                                                       'color_intensity',
                                                                       'hue',
                                                                       'dilution',
                                                                       'proline'])

    elif name == 'titanic':
        prep_path(path)
        if len(os.listdir(path_raw)) == 0:
            kaggle.api.competition_download_files('titanic', path_raw)
        titanic = pd.read_csv(os.path.join(path_raw, 'train.csv'))
        titanic_test = pd.read_csv(os.path.join(path_raw, 'test.csv'))
        return titanic, titanic_test

    elif name == 'lanl':
        prep_path(path)
        if len(os.listdir(path)) == 0:
            kaggle.api.competition_download_files('LANL-Earthquake-Prediction', path_raw)
        if not os.path.exists(os.path.join(path_raw, 'test')):
            zip_ref = zipfile.ZipFile(os.path.join(path_raw, 'test.zip'), 'r')
            zip_ref.extractall(os.path.join(path_raw, 'test'))
            zip_ref.close()
        return pd.read_csv(os.path.join(path_raw, 'train.csv.zip'))

    elif name == 'MNIST':
        mnist = torchvision.datasets.MNIST('data', train=True, download=True)
        mnist_test = torchvision.datasets.MNIST('data', train=False, download=True)
        return mnist, mnist_test

    elif name == 'FashionMNIST':
        fmnist = torchvision.datasets.FashionMNIST('data', train=True, download=True)
        fmnist_test = torchvision.datasets.FashionMNIST('data', train=False, download=True)
        return fmnist, fmnist_test


def load_processed_dataset(name):
    """
    Load desired data set into memory from processed folder
    :param name: name of data set requested
    :return: data set requested (comes in various forms based on the desired data)
    """
    assert name in VALID_NAMES, 'Invalid data set requested. Please make sure name is one of ' + ', '.join(VALID_NAMES) + '.'
    path = os.path.join('downloads', name)
    path_processed = os.path.join(path, 'processed')

    if name == 'iris':
        return pd.read_csv(os.path.join(path_processed, 'iris.csv'))

    elif name == 'wine':
        return pd.read_csv(os.path.join(path_processed, 'wine.csv'))

    elif name == 'titanic':
        return pd.read_csv(os.path.join(path_processed, 'titanic.csv'))

    elif name == 'lanl':
        with open(os.path.join(path_processed, 'train_data.pkl'), 'rb') as f:
            x = pkl.load(f)
        with open(os.path.join(path_processed, 'train_targets.pkl'), 'rb') as f:
            y = pkl.load(f)
        return x, y

    elif name == 'MNIST' or name == 'FashionMNIST':
        training = torch.load(os.path.join(path_processed, 'training.pt'))
        test = torch.load(os.path.join(path_processed, 'test.pt'))
        return training, test


def import_dataset(path, bs, shuffle):
    """
    Image generator for a directory containing folders as label names (and images of that label within each of these label-named folders)
    :param path: Path to parent directory
    :param bs: Batch size
    :param shuffle: Whether to shuffle the data order
    :return: PyTorch DataLoader
    """
    dataset = ImageFolderWithPaths(
        root=path,
        transform=torchvision.transforms.ToTensor()
    )
    loader = data.DataLoader(
        dataset,
        batch_size=bs,
        num_workers=0,
        shuffle=shuffle
    )
    return loader


def preprocess_imported_dataset(path, import_gen, splits=None, x_dim=None):
    """
    Preprocesses entire image data set, cropping images and splitting them into train and validation folders.
    Returns import information for future steps
    1. Scan data set and map it by label
    2. Split into train/val/test
    3. Encodes labels for one hot encoding
    4. Initializes directories
    5. Preprocesses images and save to directory
    6. Delete original images

    :param path: Path where unprocessed images are located and where processed images will be saved
    :param import_gen: PyTorch DataLoader with raw images
    :param splits: Train/Validation/Test Splits
    :param x_dim: Desired dimensions of image. If None, dimensions of first image are used.
    :return: Tuple of label encoder, one hot encoder, and image dimensions
    """
    if splits is None:
        splits = [0.80, 0.10, 0.10]  # Default

    assert round(sum(splits), 5) == 1.0
    assert len(splits) == 3

    # Scan data set, create table mapping it out by label
    dataset_map, labels = scan_image_dataset(path)
    train_val_map, test_map = train_test_split(dataset_map, test_size=splits[2], shuffle=True, stratify=dataset_map['label'])
    train_map, val_map = train_test_split(train_val_map, test_size=splits[1] / (splits[0]+splits[1]), stratify=train_val_map['label'])
    train_map['split'], val_map['split'], test_map['split'] = 'train', 'val', 'test'
    dataset_map = pd.concat((train_map, val_map, test_map), axis=0)
    dataset_map.sort_index(inplace=True)
    dataset_map.set_index(keys=['id', 'label'], inplace=True)

    # Set up paths for image folder
    safe_mkdir(path)
    safe_mkdir(os.path.join(path, "train"))
    safe_mkdir(os.path.join(path, "val"))
    safe_mkdir(os.path.join(path, "test"))

    for label in labels:
        safe_mkdir(os.path.join(path, "train", label))
        safe_mkdir(os.path.join(path, "val", label))
        safe_mkdir(os.path.join(path, "test", label))

    _, le, ohe = encode_y(labels)

    # Determine crop size if not given
    if x_dim is None:
        for x, _, _ in import_gen:
            x_dim = x[0].shape[-2], x[0].shape[-1]
            break

    # Determine ideal crop size based on architecture
    h_best_crop, _, _ = find_pow_2_arch(x_dim[0])
    w_best_crop, _, _ = find_pow_2_arch(x_dim[1])

    # Initialize transformer
    transformer = t.Compose([
        t.ToPILImage(),
        t.CenterCrop((x_dim[0] - h_best_crop, x_dim[1] - w_best_crop))
    ])

    # Preprocess images and save into train/val folders
    for x, y, img_ids in import_gen:
        for i in range(len(x)):
            img = transformer(x[i])
            label = le.inverse_transform(y[i].view(-1)).take(0)
            split = dataset_map.loc[img_ids[i], label].values.take(0)
            img.save(os.path.join(path, split, label, img_ids[i]))

    # Delete original images to save space
    for label in labels:
        shutil.rmtree(os.path.join(path, label))

    return le, ohe, (x_dim[0] - h_best_crop, x_dim[1] - w_best_crop)


def scan_image_dataset(path):
    """
    Loops through image data set and produces a table with info about the data set
    Assumes all extensions are the same
    :param path: Path to image data set
    :return: Tuple of table with one row per image, with file name and label as features, and a vector of labels
    """
    labels = sorted(os.listdir(path))

    dict = {'id': [], 'label': []}
    for label in labels:
        tmp_ids = os.listdir(os.path.join(path, label))
        tmp_labels = [label for id in tmp_ids]
        dict['id'] += tmp_ids
        dict['label'] += tmp_labels

    df = pd.DataFrame(data=dict)

    return df, labels


def find_img_folder_name(data_dir):
    """Loops through contents of unzipped data folder and returns the first folder it finds"""
    return [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))][0]
