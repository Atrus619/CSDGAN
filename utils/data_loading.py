import kaggle
import wget
import torchvision
import os
import pandas as pd
import zipfile


def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def safe_dl(url, path):
    if not os.path.exists(path + '/' + url.split('/')[-1]):
        wget.download(url, path)


# There are several different types of data sets included in this analysis. We will work with each one separately.
def load_dataset(name):
    """
    Checks to see if data has been installed to the data folder, and installs it if not.
    Loads into memory the desired data set.
    :param name: name of data set requested
    :return: data set requested (comes in various forms based on the desired data)
    """
    name_list = ['iris', 'wine', 'titanic', 'lanl', 'mnist', 'fmnist']
    assert name in name_list, 'Invalid data set requested. Please make sure name is one of ' + ', '.join(name_list) + '.'

    safe_mkdir('data')
    kaggle.api.authenticate()  # TODO: Ensure that kaggle authentication is defensively coded
    dir = 'data/' + name

    if name == 'iris':
        safe_mkdir(dir)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', dir)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names', dir)
        data = pd.read_csv('data/iris/iris.data', names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'species'])
    elif name == 'wine':
        safe_mkdir(dir)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', dir)
        safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names', dir)
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
        safe_mkdir(dir)
        if len(os.listdir(dir)) == 0:
            kaggle.api.competition_download_files('titanic', dir)
        titanic = pd.read_csv('data/titanic/train.csv')
        titanic_test = pd.read_csv('data/titanic/test.csv')
        data = [titanic, titanic_test]
    elif name == 'lanl':
        safe_mkdir(dir)
        if len(os.listdir(dir)) == 0:
            kaggle.api.competition_download_files('LANL-Earthquake-Prediction', dir)
        if not os.path.exists(dir + "/test"):
            zip_ref = zipfile.ZipFile(dir + "/test.zip", 'r')
            zip_ref.extractall(dir + "/test")
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

