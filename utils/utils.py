from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import wget
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os


def fix_wd():
    """Helper function to set working directory to the primary one"""
    while os.path.basename(os.getcwd()) != 'Synthetic_Data_GAN_Capstone':
        os.chdir('..')


def safe_mkdir(path):
    """Create a directory if there isn't one already"""
    try:
        os.mkdir(path)
    except OSError:
        pass


def safe_dl(url, path):
    if not os.path.exists(path + '/' + url.split('/')[-1]):
        wget.download(url, path)


def train_test_logistic_reg(x_train, y_train, x_test, y_test, param_grid, cv=5, random_state=None, labels_list=None, verbose=True):
    """
    Helper function to repeatedly test and print outputs for a logistic regression
    :param x_train: training data, NumPy array
    :param y_train: training labels, NumPy array
    :param x_test: testing data, NumPy array
    :param y_test: testing labels, NumPy array
    :param param_grid: parameter grid for GridSearchCV
    :param cv: number of folds
    :param random_state: Seed for reproducibility
    :param labels_list: List of names of labels
    :param verbose: Verbosity for whether to print all information (True = print, False = don't print)
    :return: Best fitted score
    """
    lr = LogisticRegression(penalty='elasticnet', multi_class='multinomial', solver='saga', random_state=random_state, max_iter=10000)
    lr_cv = GridSearchCV(lr, param_grid=param_grid, n_jobs=-1, cv=cv, iid=True)
    lr_cv.fit(x_train, y_train)
    best_score = lr_cv.score(x_test, y_test)
    predictions = lr_cv.predict(x_test)
    if verbose:
        print("Best Accuracy: {0:.2%}".format(best_score))
        print("Best Parameters:", lr_cv.best_params_)
        print(classification_report(y_test, predictions, labels=labels_list))
        print(confusion_matrix(np.array(y_test), predictions, labels=labels_list))
    return best_score


def plot_scatter_matrix(df, cont_inputs, title, scaler=None, show=True, save=None):
    """
    Plot scatter matrix of data set (real or fake)
    :param df: DataFrame
    :param cont_inputs: List of names of the continuous inputs
    :param title: Title to be attached to saved file
    :param scaler: Optional scaler for inverse transforming data back to original scale
    :param show: Whether to show the plot
    :param save: File path to save the resulting plot. If None, plot is not saved.
    """
    X = df[cont_inputs]

    if scaler:
        X = scaler.inverse_transform(X)
    pd.plotting.scatter_matrix(pd.DataFrame(X, columns=cont_inputs), figsize=(12, 12))

    st = plt.suptitle(title + ' Scatter Matrix', fontsize='x-large', fontweight='bold')
    plt.tight_layout()
    st.set_y(0.96)
    plt.subplots_adjust(top=0.9)

    if show:
        plt.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/scatter_matrices')
        plt.savefig(save + '/scatter_matrices/' + title + '_scatter_matrix.png')


def plot_conditional_scatter(col1, col2, real_df, fake_df, dep_var, cont_inputs, labels_list, scaler=None, alpha=1.0, show=True, save=None):
    """
    Plot conditional scatter plot (labels are colors) to compare real and fake data side by side
    :param col1: Column index of first feature to be plotted (x-axis)
    :param col2: Column index of second feature to be plotted (y-axis)
    :param real_df: Original DataFrame
    :param fake_df: Generated DataFrame
    :param dep_var: Name of dependent variable (string)
    :param cont_inputs: List of names of the continuous inputs
    :param labels_list: List of names of labels
    :param scaler: Optional scaler for inverse transforming data back to original scale
    :param alpha: Optional alpha parameter for matplotlib.pyplot
    :param show: Whether to show the plot (boolean)
    :param save: File path to save the resulting plot. If None, plot is not saved.
    """
    assert col1 in real_df.columns and col2 in real_df.columns, "Column not contained in DataFrame"

    x_real, x_fake = real_df[cont_inputs].values, fake_df[cont_inputs].values
    y_real, y_fake = real_df[dep_var].values, fake_df[dep_var].values

    col1_index = np.where(real_df[cont_inputs].columns == col1)[0].take(0)
    col2_index = np.where(real_df[cont_inputs].columns == col2)[0].take(0)

    if scaler:
        x_real = scaler.inverse_transform(x_real)
        x_fake = scaler.inverse_transform(x_fake)

    f, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    axes[0].title.set_text("Real")
    axes[1].title.set_text("Fake")

    clrs = sns.color_palette('husl', n_colors=len(labels_list))
    for i, label in enumerate(labels_list):
        axes[0].scatter(x=x_real[:, col1_index][y_real == label], y=x_real[:, col2_index][y_real == label], label=label, c=clrs[i], alpha=alpha)
        axes[1].scatter(x=x_fake[:, col1_index][y_fake == label], y=x_fake[:, col2_index][y_fake == label], label=label, c=clrs[i], alpha=alpha)

    axes[0].set_xlabel(col1)
    axes[1].set_xlabel(col1)
    axes[0].set_ylabel(col2)

    axes[1].legend()

    st = f.suptitle(col1 + " vs. " + col2 + ' Conditional Scatter Plot', fontsize='x-large', fontweight='bold')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.85)

    if show:
        f.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/conditional_scatters')
        f.savefig(save + '/conditional_scatters/' + col1 + '_vs_' + col2 + '_conditional_scatter.png')


def plot_conditional_density(col, real_df, fake_df, dep_var, cont_inputs, labels_list, scaler=None, show=True, save=None):
    """
    Plot conditional density plot (labels are colors) to compare real and fake data side by side
    :param col: Column name of feature to be plotted
    :param real_df: Original DataFrame
    :param fake_df: Generated DataFrame
    :param dep_var: Name of dependent variable (string)
    :param cont_inputs: List of names of the continuous inputs
    :param labels_list: List of names of labels
    :param scaler: Optional scaler for inverse transforming data back to original scale
    :param show: Whether to show the plot (boolean)
    :param save: File path to save the resulting plot. If None, plot is not saved.
    """
    assert col in real_df.columns, "Column not contained in DataFrame"

    x_real, x_fake = real_df[cont_inputs].values, fake_df[cont_inputs].values
    y_real, y_fake = real_df[dep_var].values, fake_df[dep_var].values

    col_index = np.where(real_df[cont_inputs].columns == col)[0].take(0)

    if scaler:
        x_real = scaler.inverse_transform(x_real)
        x_fake = scaler.inverse_transform(x_fake)

    f, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    axes[0].title.set_text("Real")
    axes[1].title.set_text("Fake")

    clrs = sns.color_palette('husl', n_colors=len(labels_list))
    for i, label in enumerate(labels_list):
        sns.distplot(a=x_real[:, col_index][y_real == label], label=label, color=clrs[i], ax=axes[0])
        sns.distplot(a=x_fake[:, col_index][y_fake == label], label=label, color=clrs[i], ax=axes[1])

    axes[1].legend()

    st = f.suptitle(col + ' Conditional Density Plot', fontsize='x-large', fontweight='bold')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.85)

    if show:
        f.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/conditional_densities')
        f.savefig(save + '/conditional_densities/' + col + '_conditional_density.png')


def scale_cont_inputs(arr, preprocessed_cat_mask, scaler=None):
    """
    Transform array's continuous features
    :param arr: NumPy array to be transformed
    :param preprocessed_cat_mask: Boolean mask of which features in array are categorical (True) versus continuous (False)
    :param scaler: Optional scaler. If not provided, one will be created.
    :return: NumPy array with continuous features scaled and categorical features left alone, along with the scaler used for transformation
    """
    if scaler is None:
        scaler = StandardScaler()
        arr_cont = scaler.fit_transform(arr[:, ~preprocessed_cat_mask])
    else:
        arr_cont = scaler.transform(arr[:, ~preprocessed_cat_mask])
    arr_cat = arr[:, preprocessed_cat_mask]
    return np.concatenate((arr_cat, arr_cont), axis=1), scaler


def encode_categoricals_custom(df, x_train, x_test, cat_inputs, cat_mask):
    """
    Generate a two-way conversion of one_hot encoding categorical variables
    :param df: Raw DataFrame for encoding
    :param x_train: Subset of raw DataFrame to be used as training data
    :param x_test: Subset of raw DataFrame to be used as testing data
    :param cat_inputs: List of names of features in df that are categorical
    :param cat_mask: Boolean mask of which features in one hot encoded version of DataFrame are categorical (True) versus continuous (False)
    :return: Dictionary of LabelEncoders to be used for inverse transformation back to original raw data, OneHotEncoder for same purpose, and transformed train and test data
    """
    le_dict = {}
    for x in cat_inputs:
        le_dict[x] = LabelEncoder()
        le_dict[x] = le_dict[x].fit(df[x])
        x_train[x] = le_dict[x].transform(x_train[x])
        x_test[x] = le_dict[x].transform(x_test[x])

    ohe = OneHotEncoder(categorical_features=cat_mask, sparse=False)
    ohe.fit(pd.concat([x_train, x_test], axis=0))
    x_train = ohe.transform(x_train)
    x_test = ohe.transform(x_test)
    return le_dict, ohe, x_train, x_test


def create_preprocessed_cat_mask(le_dict, x_train):
    """
    :param le_dict: Dictionary of LabelEncoders to be used for inverse transformation back to original raw data
    :param x_train: Training data
    :return: Boolean mask of which features in array are categorical (True) versus continuous (False)
    """
    count = 0
    for _, le in le_dict.items():
        count += len(le.classes_)
    return np.concatenate([np.full(count, True), np.full(x_train.shape[1] - count, False)])


def compare_cats(real_df, fake_df, x, y, hue, show=True, save=None):
    """
    Visualize categorical variables
    :param real_df: DataFrame of original raw data
    :param fake_df: DataFrame of generated data. Output of fully_process_fake_output method.
    :param x: Name of first feature to be compared (str)
    :param y: Name of second feature to be compared (str)
    :param hue: Name of third feature to be compared (str)
    :param show: Whether to show the resulting plot
    :param save: File path to save the resulting plot. If None, plot is not saved.
    """
    f, axes = plt.subplots(1, 2, figsize=(8, 8), sharey=True, sharex=True)

    axes[0].title.set_text('Fake Data')
    axes[1].title.set_text('Real Data')

    sns.catplot(data=real_df, x=x, y=y, hue=hue, kind='bar', ax=axes[0])
    sns.catplot(data=fake_df, x=x, y=y, hue=hue, kind='bar', ax=axes[1])

    sup = "Comparing {0} by {1} and {2}".format(y, x, hue)
    st = f.suptitle(sup, fontsize='x-large', fontweight='bold')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)

    # Quick n' dirty solution to sns.catplot creating its own blank subplots
    plt.close(2)
    plt.close(3)

    if show:
        f.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/compare_cats')
        f.savefig(save + '/compare_cats/' + x + '_' + hue + '_cat_comparison.png')


def train_val_test_split(x, y, splits, random_state=None):
    """
    Performs a train/validation/test split on x and y, stratified, based on desired splits
    :param x: Independent var
    :param y: Dependent var
    :param splits: Proportion of total data to be assigned to train/validation/test set
    :param random_state: Optional random state
    :return: Data split into 6 discrete pieces
    """
    assert sum(splits) == 1.0, "Please make sure sum of splits is equal to 1"
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=splits[2], stratify=y, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=splits[1] / (splits[0]+splits[1]), stratify=y_train, random_state=random_state)
    x_train, y_train, x_val, y_val, x_test, y_test = torch.from_numpy(x_train), torch.from_numpy(y_train), \
                                                     torch.from_numpy(x_val), torch.from_numpy(y_val), \
                                                     torch.from_numpy(x_test), torch.from_numpy(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


def convert_np_hist_to_plot(np_hist):
    """
    Converts a numpy representation of a histogram into a matplotlib.pyplot object
    :param np_hist: tuple generated by np.histogram(vec)
    :return: Returns a matplotlib.pyplot bar plot object
    """
    height, bins = np_hist
    width = (bins.max() - bins.min())/(len(bins) - 1)
    return plt.bar(height=height, x=bins[:-1], width=width)


def define_cat_inputs(df, dep_var, cont_inputs):
    """
    Defines a list of cat_inputs and a boolean cat_mask based on a df and list of cont_inputs
    :param df: DataFrame of interest
    :param dep_var: String of name of dependent variable column. Excluded from consideration.
    :param cont_inputs: List of strings of names of continuous feature columns
    :return: Tuple of list of strings of names of implied categorical feature columns, and a boolean mask (NumPy array) for whether the feature is categorical
    """
    cat_inputs = [x for x in df.drop(columns=dep_var).columns if x not in cont_inputs]
    cat_mask = ~np.array([x in cont_inputs for x in df.drop(columns=dep_var).columns])
    return cat_inputs, cat_mask


def reorder_cols(df, dep_var, cont_inputs):
    """Reorders columns in DataFrame so that categorical inputs are first"""
    cols = df.columns.tolist()
    cat_inputs = [x for x in df.drop(columns=dep_var).columns if x not in cont_inputs]
    cols_start = [dep_var] + cat_inputs
    cols = cols_start + [x for x in cols if x not in cols_start]
    return df[cols]


def print_tabular_config(cfg):
    print("Experiment Name:", cfg.EXPERIMENT_NAME)  # Where output files will be saved
    print("Manual Seed:", cfg.MANUAL_SEED)  # This is above

    print("\nEvaluation Parameters:")
    print("Number of data set pass-throughs per epoch:", cfg.CADENCE)
    print("Continuous Features:", cfg.CONT_INPUTS)
    print("Integer Features:", cfg.INT_INPUTS)
    print("Dependent Variable:", cfg.DEP_VAR)
    print("Evaluation Parameter Grid:", cfg.EVAL_PARAM_GRID)
    print("Number of cross-validation folds for evaluation:", cfg.EVAL_FOLDS)
    print("Generated data set sizes to test for during evaluation:", cfg.TEST_RANGES)

    print("\nTraining Parameters:")  # Whether to shuffle the data for training, and number of cpu workers for concurrency
    print("Test Set Size:", cfg.TEST_SIZE)  # Number of examples in the test set
    print("Total Number of Epochs:", cfg.NUM_EPOCHS)
    print("Printing Frequency:", cfg.PRINT_FREQ)  # How often to print results to the console (in epochs)
    print("Evaluation Frequency:", cfg.EVAL_FREQ)  # How often to train the evaluator on generated data (in epochs)

    for key, value in cfg.TRAINING_PARAMS.items():
        print(key + ": " + str(value))

    print("\nCGAN Class Initialization Parameters:")  # See the config file for more detail behind what these choices represent
    for key, value in cfg.CGAN_INIT_PARAMS.items():
        print(key + ": " + str(value))


def print_dc_config(cfg):
    print("Experiment Name:", cfg.EXPERIMENT_NAME)  # Where output files will be saved
    print("Manual Seed:", cfg.MANUAL_SEED)  # This is above

    print("\nEvaluation Parameters:")
    print("Train/Test/Validation Splits:", cfg.SPLITS)  # Must sum to 1.0
    print("Total Number of Epochs:", cfg.NUM_EPOCHS)
    print("Printing Frequency:", cfg.PRINT_FREQ)  # How often to print results to the console (in epochs)
    print("Evaluation Frequency:", cfg.EVAL_FREQ)  # How often to train the evaluator on generated data (in epochs)

    print("\nTraining Parameters:")  # Whether to shuffle the data for training, and number of cpu workers for concurrency
    for key, value in cfg.TRAINING_PARAMS.items():
        print(key + ": " + str(value))

    print("\nCGAN Class Initialization Parameters:")  # See the config file for more detail behind what these choices represent
    for key, value in cfg.CGAN_INIT_PARAMS.items():
        print(key + ": " + str(value))