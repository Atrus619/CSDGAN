import kaggle
import torchvision
import os
import pandas as pd
import zipfile
from utils.utils import safe_dl
import pickle as pkl
import torch

VALID_NAMES = {'iris', 'wine', 'titanic', 'lanl', 'MNIST', 'FashionMNIST'}


def prep_path(path):
    """Helper function to utilize safe_mkdir to safely create necessary directories for downloading data"""
    os.makedirs(os.path.join(path, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(path, 'raw'), exist_ok=True)


def load_raw_dataset(name):
    """
    Check to see if data has been installed to the downloads/raw folder, and install if not.
    Load into memory the desired data set.
    For the LANL data set, it is required to have kaggle authentication set up on your computer.
    :param name: name of data set requested
    :return: data set requested (comes in various forms based on the desired data)
    """
    assert name in VALID_NAMES, 'Invalid data set requested. Please make sure name is one of ' + ', '.join(VALID_NAMES) + '.'

    os.makedirs('downloads', exist_ok=True)
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
        mnist = torchvision.datasets.MNIST('downloads', train=True, download=True)
        mnist_test = torchvision.datasets.MNIST('downloads', train=False, download=True)
        return mnist, mnist_test

    elif name == 'FashionMNIST':
        fmnist = torchvision.datasets.FashionMNIST('downloads', train=True, download=True)
        fmnist_test = torchvision.datasets.FashionMNIST('downloads', train=False, download=True)
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
