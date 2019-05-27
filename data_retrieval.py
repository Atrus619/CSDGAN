import kaggle
import wget
import torchvision
import os


# Facilitates loading of data from various sources on the internet
def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


safe_mkdir('data')
kaggle.api.authenticate()

# Titanic data set
safe_mkdir('data/titanic')
kaggle.api.competition_download_files('titanic', 'data/titanic')

# Iris data set
safe_mkdir('data/iris')
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
wget.download(url, 'data/iris')

# Wine data set
safe_mkdir('data/wine')
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
wget.download(url, 'data/wine')

# Earthquake-LANL data set
safe_mkdir('data/Earthquake-LANL')
kaggle.api.competition_download_files('LANL-Earthquake-Prediction', 'data/Earthquake-LANL')

# MNIST data set
torchvision.datasets.MNIST('data', train=True, download=True)
torchvision.datasets.MNIST('data', train=False, download=True)

# Fashion-MNIST data set
torchvision.datasets.FashionMNIST('data', train=True, download=True)
torchvision.datasets.FashionMNIST('data', train=False, download=True)
