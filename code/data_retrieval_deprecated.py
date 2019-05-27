import kaggle
import wget
import torchvision
import os


# Facilitates loading of data from various sources on the internet
# TODO: Enable choice of which data sets to load
def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def safe_dl(url, path):
    if not os.path.exists(path + '/' + url.split('/')[-1]):
        wget.download(url, path)


safe_mkdir('data')
kaggle.api.authenticate()

# Titanic data set
titanic_dir = 'data/titanic'
safe_mkdir(titanic_dir)
if len(os.listdir(titanic_dir)) == 0:
    kaggle.api.competition_download_files('titanic', titanic_dir)

# Iris data set
iris_dir = 'data/iris'
safe_mkdir(iris_dir)
safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', iris_dir)
safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names', iris_dir)

# Wine data set
wine_dir = 'data/wine'
safe_mkdir(wine_dir)
safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', wine_dir)
safe_dl('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names', wine_dir)

# Earthquake-LANL data set
LANL_dir = 'data/Earthquake-LANL'
safe_mkdir(LANL_dir)
if len(os.listdir(LANL_dir)) == 0:
    kaggle.api.competition_download_files('LANL-Earthquake-Prediction', LANL_dir)

# MNIST data set
torchvision.datasets.MNIST('data', train=True, download=True)
torchvision.datasets.MNIST('data', train=False, download=True)

# Fashion-MNIST data set
torchvision.datasets.FashionMNIST('data', train=True, download=True)
torchvision.datasets.FashionMNIST('data', train=False, download=True)
