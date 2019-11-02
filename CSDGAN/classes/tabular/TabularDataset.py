from torch.utils import data
import utils.utils as uu
import torch
import pandas as pd


class TabularDataset(data.Dataset):
    def __init__(self, df, dep_var, cont_inputs, int_inputs, test_size, seed=None):
        """
        Generates train/test and arr/tensor versions of the data.
        Input data is raw.
        After init, the data is scaled and transformed.
        :param df: Original raw DataFrame
        :param dep_var: Name of the dependent variable
        :param cont_inputs: List of strings of names of continuous features
        :param int_inputs: List of strings of names of integer features
        :param test_size: Size of test set (number of rows)
        :param seed: Random seed for reproducibility
        """
        self.dep_var = dep_var
        self.cont_inputs = cont_inputs
        self.int_inputs = int_inputs
        self.labels_list = list(df[dep_var].unique())

        self.df_dtypes = df.dtypes
        self.df_cols = df.columns

        # Reorganize data set
        df = uu.reorder_cols(df=df, dep_var=dep_var, cont_inputs=self.cont_inputs)
        self.cat_inputs, self.cat_mask = uu.define_cat_inputs(df=df, dep_var=dep_var, cont_inputs=cont_inputs)

        # Split data into train/test
        x_train_arr, x_test_arr, y_train_arr, y_test_arr = uu.train_test_split(df.drop(columns=dep_var), df[dep_var],
                                                                               test_size=test_size,
                                                                               stratify=df[dep_var],
                                                                               random_state=seed)

        # Convert all categorical variables to dummies, and save two-way transformation
        self.le_dict, self.ohe, x_train_arr, x_test_arr = uu.encode_categoricals_custom(df=df,
                                                                                        x_train=x_train_arr,
                                                                                        x_test=x_test_arr,
                                                                                        cat_inputs=self.cat_inputs,
                                                                                        cat_mask=self.cat_mask)
        self.preprocessed_cat_mask = uu.create_preprocessed_cat_mask(le_dict=self.le_dict, x_train=x_train_arr)

        # Scale continuous inputs
        if len(self.cont_inputs) == 0:
            self.scaler = None
        else:
            x_train_arr, self.scaler = uu.scale_cont_inputs(arr=x_train_arr,
                                                            preprocessed_cat_mask=self.preprocessed_cat_mask)
            x_test_arr, _ = uu.scale_cont_inputs(arr=x_test_arr, preprocessed_cat_mask=self.preprocessed_cat_mask,
                                                 scaler=self.scaler)

        # Convert to tensor-friendly format
        self.x_train, self.x_test, self.y_train, self.y_test = self.preprocess_data(x_train_arr=x_train_arr,
                                                                                    y_train_arr=y_train_arr,
                                                                                    x_test_arr=x_test_arr,
                                                                                    y_test_arr=y_test_arr)
        self.out_dim = self.x_train.shape[1]
        self.eval_stratify = list(self.y_train.mean(0).detach().cpu().numpy())

        # Set current device
        self.device = self.get_dev()

    def preprocess_data(self, x_train_arr, y_train_arr, x_test_arr, y_test_arr):
        """Converts input arrays of data into tensors ready for training"""
        x_train = torch.tensor(x_train_arr, dtype=torch.float)

        x_test = torch.tensor(x_test_arr, dtype=torch.float)

        y_train_dummies = pd.get_dummies(y_train_arr)
        y_train = torch.tensor(y_train_dummies.values, dtype=torch.float)

        y_test_dummies = pd.get_dummies(y_test_arr)
        y_test = torch.tensor(y_test_dummies.values, dtype=torch.float)

        return x_train, x_test, y_train, y_test

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def to_dev(self, device):
        """Moves entire data set to specified device. Can be helpful in speeding up training times for small data sets (~60-100x improvement in speed)."""
        self.x_train, self.y_train, self.x_test, self.y_test = self.x_train.to(device), self.y_train.to(
            device), self.x_test.to(device), self.y_test.to(device)
        self.device = device

    def get_dev(self):
        return self.x_train.device
