import kaggle
import torchvision
import os
import pandas as pd
import zipfile
from utils.utils import safe_mkdir, safe_dl


def prep_path(path):
    """Helper function to utilize safe_mkdir to safely create necessary directories for downloading data"""
    safe_mkdir(path)
    safe_mkdir(path + '/processed')
    safe_mkdir(path + '/raw')


def load_dataset(name):
    """
    Checks to see if data has been installed to the data folder, and installs it if not.
    Loads into memory the desired data set.
    For the LANL data set, it is required to have kaggle authentication set up on your computer.
    :param name: name of data set requested
    :return: data set requested (comes in various forms based on the desired data)
    """
    valid_names = {'iris', 'wine', 'titanic', 'lanl', 'mnist', 'fmnist'}
    assert name in valid_names, 'Invalid data set requested. Please make sure name is one of ' + ', '.join(valid_names) + '.'

    safe_mkdir('downloads')
    kaggle.api.authenticate()
    path = 'downloads/' + name

    if name == 'iris':
        prep_path(path)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', path)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names', path)
        data = pd.read_csv('data/iris/iris.data', names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'species'])

    elif name == 'wine':
        prep_path(path)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', path)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names', path)
        data = pd.read_csv('data/wine/wine.data', names=['class',
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
        if len(os.listdir(path)) == 0:
            kaggle.api.competition_download_files('titanic', path)
        titanic = pd.read_csv('data/titanic/train.csv')
        titanic_test = pd.read_csv('data/titanic/test.csv')
        data = [titanic, titanic_test]

    elif name == 'lanl':
        prep_path(path)
        if len(os.listdir(path)) == 0:
            kaggle.api.competition_download_files('LANL-Earthquake-Prediction', path)
        if not os.path.exists(path + "/test"):
            zip_ref = zipfile.ZipFile(path + "/test.zip", 'r')
            zip_ref.extractall(path + "/test")
            zip_ref.close()
        data = pd.read_csv('data/lanl/train.csv.zip')

    elif name == 'mnist':
        mnist = torchvision.datasets.MNIST('data', train=True, download=True)
        mnist_test = torchvision.datasets.MNIST('data', train=False, download=True)
        data = [mnist, mnist_test]

    elif name == 'fmnist':
        fmnist = torchvision.datasets.FashionMNIST('data', train=True, download=True)
        fmnist_test = torchvision.datasets.FashionMNIST('data', train=False, download=True)
        data = [fmnist, fmnist_test]

    return data
