# TODO: Add argparser to specify data set and call data_loading.py functions
import pandas as pd
from utils.data_loading import load_raw_dataset

wine = load_raw_dataset('wine')
wine.to_csv('downloads/wine/processed/wine.csv', index=False)

iris = load_raw_dataset('iris')
iris.to_csv('downloads/iris/processed/iris.csv', index=False)
