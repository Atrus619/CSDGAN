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
import random


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


# def gen_fake_data(netG, bs, nz, nc, labels_list, device, stratify=None):
#     """
#     Generate fake data. Calls gen_labels method below.
#     :param netG: netG class to use
#     :param bs: Batch size of fake data to generate
#     :param nz: Size of noise vector for netG
#     :param nc: Number of classes
#     :param labels_list: List of names of each of the classes
#     :param device: Device (either gpu or cpu)
#     :param stratify: Proportions of each label to stratify
#     :return: Tuple of generated data and associated labels
#     """
#     noise = torch.randn(bs, nz, device=device)
#     fake_labels, output_labels = gen_labels(size=bs, num_classes=nc, labels_list=labels_list, stratify=stratify)
#     fake_labels = fake_labels.to(device)
#     fake_data = netG(noise, fake_labels).cpu().detach().numpy()
#     return fake_data, output_labels


# def gen_labels(size, num_classes, labels_list, stratify=None):
#     """
#     Generate labels for generating fake data
#     :param size: Number of desired labels
#     :param num_classes: Number of classes
#     :param labels_list: List of names of each of the classes
#     :param stratify: Proportions of each label to stratify
#     :return: Tuple of one hot encoded labels and the labels themselves
#     """
#     if stratify is None:
#         assert size // num_classes == size / num_classes, "Make sure size is divisible by num_classes"
#         stratify = np.full(num_classes, 1 / num_classes)
#     else:
#         assert np.sum(stratify) == 1, "Make sure stratify sums to 1"
#     counts = np.round(np.dot(stratify, size), decimals=0).astype('int')
#     while np.sum(counts) != size:
#         if np.sum(counts) > size:
#             counts[random.choice(range(num_classes))] -= 1
#         else:
#             counts[random.choice(range(num_classes))] += 1
#     output_one_hot = np.empty((0, num_classes))
#     one_hot = pd.get_dummies(labels_list)
#     output_labels = np.empty(0)
#     for i in range(num_classes):
#         tmp_one_hot = np.empty((counts[i], num_classes))
#         tmp_labels = np.full(counts[i], labels_list[i])
#         output_labels = np.concatenate((output_labels, tmp_labels), axis=0)
#         for j in range(num_classes):
#             tmp_one_hot[:, j] = one_hot.iloc[i, j]
#         output_one_hot = np.concatenate((output_one_hot, tmp_one_hot), axis=0)
#         output_one_hot = torch.tensor(output_one_hot, dtype=torch.float)
#     return output_one_hot, output_labels


# def evaluate_training_progress(test_range, fake_bs, nz, nc, out_dim, netG, x_test, y_test, manualSeed, labels_list, param_grid, device, le_dict=None, stratify=None):
#     """
#     Train a model on fake data and evaluate on test data in order to evaluate network as it trains
#     :param test_range: List of sample sizes to test
#     :param fake_bs: Batch size for generated data
#     :param nz: Size of noise vector for netG
#     :param nc: Number of classes
#     :param out_dim: Size of vector for real data (number of features)
#     :param netG: netG class to use
#     :param x_test: Testing data
#     :param y_test: Testing labels
#     :param manualSeed: Seed for reproducibility
#     :param labels_list: List of names of each of the classes
#     :param param_grid: Grid to perform GridSearchCV when training evaluation classes
#     :param device: Device (either gpu or cpu)
#     :param le_dict: Dictionary of pretrained LabelEncoders for handling categorical variables
#     :param stratify: Proportions of each label to stratify
#     :return: Tuple of list of classes trained and the scores each achieved
#     """
#     fake_scores = []
#     fake_models = []
#     for size in test_range:
#         genned_data = np.empty((0, out_dim))
#         genned_labels = np.empty(0)
#         rem = size
#         while rem > 0:
#             curr_size = min(fake_bs, rem)
#             fake_data, output_labels = gen_fake_data(netG=netG, bs=curr_size, nz=nz, nc=nc, labels_list=labels_list, device=device, stratify=stratify)
#             rem -= curr_size
#             genned_data = np.concatenate((genned_data, fake_data))
#             genned_labels = np.concatenate((genned_labels, output_labels))
#         if le_dict is not None:
#             genned_data = process_fake_output(genned_data, le_dict)
#         model_fake_tmp, score_fake_tmp = train_test_logistic_reg(x_train=genned_data, y_train=genned_labels, x_test=x_test, y_test=y_test,
#                                                                  param_grid=param_grid, cv=5, random_state=manualSeed, labels=labels_list, verbose=0)
#         fake_models.append(model_fake_tmp)
#         fake_scores.append(score_fake_tmp)
#     return fake_models, fake_scores


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


# def plot_training_progress(stored_scores, test_range, num_saves, real_data_score, save=None):
#     """
#     Plot scores of each evaluation model across training of CGAN
#     :param stored_scores: List of scores on generated data
#     :param test_range: List of sample sizes for testing
#     :param num_saves: Number of distinct checkpoints used (should be len(stored_scores) // len(test_range)
#     :param real_data_score: Best score for model trained on real data
#     :param save: File path to save the resulting plot. If None, plot is not saved
#     """
#     ys = np.empty((num_saves, len(test_range)))
#     xs = np.empty((num_saves, len(test_range)))
#     barWidth = 1 / (len(test_range) + 1)
#     for i in range(len(test_range)):
#         ys[:, i] = np.array(stored_scores[i:num_saves * len(test_range):len(test_range)])
#         xs[:, i] = np.arange(num_saves) + barWidth * i
#         plt.bar(xs[:, i], ys[:, i], width=barWidth, edgecolor='white', label=test_range[i])
#
#     line_len = 2 * len(test_range) + 1
#     plt.plot(np.linspace(0, line_len, line_len), np.full(line_len, real_data_score), linestyle='dashed', color='r')
#     plt.xlabel('Epoch', fontweight='bold')
#     plt.xticks([r + barWidth for r in range(num_saves)], list(range(num_saves)))
#     plt.title('Evaluation Over Training Epochs')
#     plt.legend(loc=4)
#     plt.show()
#
#     if save is not None:
#         assert os.path.exists(save), "Check that the desired save path exists."
#         safe_mkdir(save + '/training_progress')
#         plt.savefig(save + '/training_progress/' + 'training_progress.png')


# def parse_models(stored_models, epoch, print_interval, test_range, ind, x_test, y_test, labels):
#     """
#     Helper/diagnostic function to return stats for a specific model
#     :param stored_models: List of stored classes
#     :param epoch: Epoch of model desired
#     :param print_interval: Print interval used for training
#     :param test_range: List of sample sizes for testing
#     :param ind: Index of test_range desired
#     :param x_test: Testing data
#     :param y_test: Testing labels
#     :param labels: Names of labels
#     :return: N/A. Prints classification statistics for desired model
#     """
#     tmp_model = stored_models[epoch // print_interval * len(test_range) - 1 + ind]
#     best_score = tmp_model.score(x_test, y_test)
#     predictions = tmp_model.predict(x_test)
#     print("Accuracy:", best_score)
#     print("Best Parameters:", tmp_model.best_params_)
#     print(classification_report(y_test, predictions, labels=labels))
#     print(confusion_matrix(y_test, predictions, labels=labels))


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


def plot_conditional_scatter(col1, col2, real_df, fake_df, dep_var, cont_inputs, class_dict, scaler=None, alpha=1.0, show=True, save=None):
    """
    Plot conditional scatter plot (labels are colors) to compare real and fake data side by side
    :param col1: Column index of first feature to be plotted (x-axis)
    :param col2: Column index of second feature to be plotted (y-axis)
    :param real_df: Original DataFrame
    :param fake_df: Generated DataFrame
    :param dep_var: Name of dependent variable (string)
    :param cont_inputs: List of names of the continuous inputs
    :param class_dict: Dictionary with keys as label value (0 or 1 for binary dep var), and values as tuple of (name, 'color')
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

    for label in class_dict:
        axes[0].scatter(x=x_real[:, col1_index][y_real == label], y=x_real[:, col2_index][y_real == label], label=class_dict[label][0], c=class_dict[label][1], alpha=alpha)
        axes[1].scatter(x=x_fake[:, col1_index][y_fake == label], y=x_fake[:, col2_index][y_fake == label], label=class_dict[label][0], c=class_dict[label][1], alpha=alpha)

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


def plot_conditional_density(col, real_df, fake_df, dep_var, cont_inputs, class_dict, scaler=None, show=True, save=None):
    """
    Plot conditional density plot (labels are colors) to compare real and fake data side by side
    :param col: Column name of feature to be plotted
    :param real_df: Original DataFrame
    :param fake_df: Generated DataFrame
    :param dep_var: Name of dependent variable (string)
    :param cont_inputs: List of names of the continuous inputs
    :param class_dict: Dictionary with keys as label value (0 or 1 for binary dep var), and values as tuple of (name, 'color')
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

    for label in class_dict:
        sns.distplot(a=x_real[:, col_index][y_real == label], label=class_dict[label][0], color=class_dict[label][1], ax=axes[0])
        sns.distplot(a=x_fake[:, col_index][y_fake == label], label=class_dict[label][0], color=class_dict[label][1], ax=axes[1])

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


# def iris_plot_scatters(X, y, title, scaler=None, alpha=1.0, save=None):
#     """
#     Helper to plot iris sepal length vs width
#     :param X: DataFrame of continuous features
#     :param y: NumPy array of labels
#     :param title: Title to be attached to save file
#     :param scaler: Optional scaler for inverse transforming data back to original scale
#     :param alpha: Optional alpha parameter for matplotlib.pyplot
#     :param save: File path to save the resulting plot. If None, plot is not saved.
#     """
#     if scaler:
#         X = scaler.inverse_transform(X)
#     inv = pd.DataFrame(X).rename(columns={0: 'sepal_len', 1: 'sepal_wid', 2: 'petal_len', 3: 'petal_wid'})
#     inv = pd.concat((inv, pd.DataFrame(y)), axis=1).rename(columns={0: 'species'})
#
#     fig = plt.figure()
#
#     plt.subplot(1, 2, 1)
#     plt.scatter(inv.sepal_len[inv.species == 'Iris-setosa'], inv.sepal_wid[inv.species == 'Iris-setosa'], c='r', label='setosa', alpha=alpha)
#     plt.scatter(inv.sepal_len[inv.species == 'Iris-versicolor'], inv.sepal_wid[inv.species == 'Iris-versicolor'], c='b', label='versicolor', alpha=alpha)
#     plt.scatter(inv.sepal_len[inv.species == 'Iris-virginica'], inv.sepal_wid[inv.species == 'Iris-virginica'], c='g', label='virginica', alpha=alpha)
#     # plt.legend()
#     plt.xlabel('sepal length')
#     plt.ylabel('sepal width')
#     plt.title('Sepal Length vs Width')
#
#     plt.subplot(1, 2, 2)
#     plt.scatter(inv.petal_len[inv.species == 'Iris-setosa'], inv.petal_wid[inv.species == 'Iris-setosa'], c='r', label='setosa', alpha=alpha)
#     plt.scatter(inv.petal_len[inv.species == 'Iris-versicolor'], inv.petal_wid[inv.species == 'Iris-versicolor'], c='b', label='versicolor', alpha=alpha)
#     plt.scatter(inv.petal_len[inv.species == 'Iris-virginica'], inv.petal_wid[inv.species == 'Iris-virginica'], c='g', label='virginica', alpha=alpha)
#
#     plt.legend()
#     plt.xlabel('petal length')
#     plt.ylabel('petal width')
#     plt.title('Petal Length vs Width')
#
#     st = plt.suptitle(title + " Feature Scatter Plots", fontsize='x-large')
#     plt.tight_layout()
#
#     # shift subplots down:
#     st.set_y(0.95)
#     fig.subplots_adjust(top=0.85)
#
#     plt.show()
#
#     if save is not None:
#         assert os.path.exists(save), "Check that the desired save path exists."
#         safe_mkdir(save + '/conditional_scatters')
#         plt.savefig(save + '/conditional_scatters/' + title + '_conditional_scatter.png')
#
#
# def iris_plot_densities(X, y, title, scaler=None, save=None):
#     """
#     Helper to plot iris distributions of variables by class
#     :param X: DataFrame of continuous features
#     :param y: NumPy array of labels
#     :param title: Title to be attached to save file
#     :param scaler: Optional scaler for inverse transforming data back to original scale
#     :param save: File path to save the resulting plot. If None, plot is not saved.
#     """
#     if scaler:
#         X = scaler.inverse_transform(X)
#     inv = pd.DataFrame(X).rename(columns={0: 'sepal_len', 1: 'sepal_wid', 2: 'petal_len', 3: 'petal_wid'})
#     inv = pd.concat((inv, pd.DataFrame(y)), axis=1).rename(columns={0: 'species'})
#
#     f, axes = plt.subplots(3, 4, figsize=(8, 8), sharex=True, sharey=True)
#
#     sns.distplot(inv['sepal_len'][inv.species == 'Iris-setosa'], color='red', ax=axes[0, 0])
#     sns.distplot(inv['sepal_len'][inv.species == 'Iris-versicolor'], color='blue', ax=axes[1, 0])
#     sns.distplot(inv['sepal_len'][inv.species == 'Iris-virginica'], color='green', ax=axes[2, 0])
#     sns.distplot(inv['sepal_wid'][inv.species == 'Iris-setosa'], color='red', ax=axes[0, 1])
#     sns.distplot(inv['sepal_wid'][inv.species == 'Iris-versicolor'], color='blue', ax=axes[1, 1])
#     sns.distplot(inv['sepal_wid'][inv.species == 'Iris-virginica'], color='green', ax=axes[2, 1])
#     sns.distplot(inv['petal_len'][inv.species == 'Iris-setosa'], color='red', ax=axes[0, 2])
#     sns.distplot(inv['petal_len'][inv.species == 'Iris-versicolor'], color='blue', ax=axes[1, 2])
#     sns.distplot(inv['petal_len'][inv.species == 'Iris-virginica'], color='green', ax=axes[2, 2])
#     sns.distplot(inv['petal_wid'][inv.species == 'Iris-setosa'], color='red', ax=axes[0, 3])
#     sns.distplot(inv['petal_wid'][inv.species == 'Iris-versicolor'], color='blue', ax=axes[1, 3])
#     sns.distplot(inv['petal_wid'][inv.species == 'Iris-virginica'], color='green', ax=axes[2, 3])
#
#     axes[0, 0].set_ylabel('Iris-setosa')
#     axes[1, 0].set_ylabel('Iris-versicolor')
#     axes[2, 0].set_ylabel('Iris-virginica')
#
#     plt.xlim(0, 10)
#     plt.ylim(0, 2)
#
#     plt.xticks(list(range(11)))
#     plt.yticks(list(np.linspace(0, 2, 11)))
#
#     st = plt.suptitle(title + " Feature Density Plots", fontsize='x-large')
#     plt.tight_layout()
#
#     # shift subplots down:
#     st.set_y(.95)
#     f.subplots_adjust(top=0.9)
#
#     f.show()
#
#     if save is not None:
#         assert os.path.exists(save), "Check that the desired save path exists."
#         safe_mkdir(save + '/conditional_densities')
#         f.savefig(save + '/conditional_densities/' + title + '_conditional_density.png')


# def training_plots(netD, netG, num_epochs, save=None):
#     """
#     Pull together a plot of relevant training diagnostics for both netG and netD
#     :param netD: Class netD
#     :param netG: Class netG
#     :param num_epochs: Number of epochs trained for
#     :param save: File path to save the resulting plot. If None, plot is not saved.
#     """
#     f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True)
#
#     axes[0, 0].title.set_text("Generator and Discriminator Loss During Training")
#     axes[0, 0].plot(netG.losses, label="G")
#     axes[0, 0].plot(netD.losses, label="D")
#     axes[0, 0].set_xlabel("iterations")
#     axes[0, 0].set_ylabel("loss")
#     axes[0, 0].legend()
#
#     axes[0, 1].title.set_text("Average Discriminator Outputs During Training")
#     axes[0, 1].plot(netD.Avg_D_reals, label="Real")
#     axes[0, 1].plot(netD.Avg_D_fakes, label="Fake")
#     axes[0, 1].plot(np.linspace(0, num_epochs, num_epochs), np.full(num_epochs, 0.5))
#     axes[0, 1].set_xlabel("iterations")
#     axes[0, 1].set_ylabel("proportion")
#     axes[0, 1].legend()
#
#     axes[1, 0].title.set_text('Gradient Norm History')
#     axes[1, 0].plot(netG.gnorm_total_history, label="G")
#     axes[1, 0].plot(netD.gnorm_total_history, label="D")
#     axes[1, 0].set_xlabel("iterations")
#     axes[1, 0].set_ylabel("norm")
#     axes[1, 0].legend()
#
#     axes[1, 1].title.set_text('Weight Norm History')
#     axes[1, 1].plot(netG.wnorm_total_history, label="G")
#     axes[1, 1].plot(netD.wnorm_total_history, label="D")
#     axes[1, 1].set_xlabel("iterations")
#     axes[1, 1].set_ylabel("norm")
#     axes[1, 1].legend()
#
#     st = f.suptitle("Training Diagnostic Plots", fontsize='x-large')
#     f.tight_layout()
#     st.set_y(0.96)
#     f.subplots_adjust(top=0.9)
#
#     f.show()
#
#     if save is not None:
#         assert os.path.exists(save), "Check that the desired save path exists."
#         safe_mkdir(save + '/training_plots')
#         f.savefig(save + '/training_plots/training_plot.png')


# def fake_data_training_plots(real_range, score_real, test_range, fake_scores, save=None):
#     """
#     Plots of evaluation progress across epochs
#     Only used for iris data set
#     :param real_range: Number of samples used for real data evaluation
#     :param score_real: Real data evaluation best score
#     :param test_range: List of number of samples used for fake data evaluation
#     :param fake_scores: List of best scores for each fake data evaluation
#     :param save: File path to save the resulting plot. If None, plot is not saved.
#     """
#     f, axes = plt.subplots(1, 2, figsize=(8, 8))
#
#     axes[0].title.set_text('Sample Sizes')
#     labels = ['real' + str(real_range)] + ['fake' + str(x) for x in test_range]
#     axes[0].bar(x=labels, height=[real_range] + test_range)
#     axes[0].set_xticklabels(labels=labels, rotation='vertical')
#
#     axes[1].title.set_text('OOF Accuracy')
#     axes[1].bar(x=labels, height=[score_real] + fake_scores)
#     axes[1].set_xticklabels(labels=labels, rotation='vertical')
#     axes[1].set_ylim(0.8, 1.0)
#
#     st = f.suptitle("Fake Data Training Results", fontsize='x-large')
#     f.tight_layout()
#     st.set_y(0.96)
#     f.subplots_adjust(top=0.9)
#
#     f.show()
#
#     if save is not None:
#         assert os.path.exists(save), "Check that the desired save path exists."
#         safe_mkdir(save + '/fake_data_training_plots')
#         f.savefig(save + '/fake_data_training_plots/fake_data_training_plot.png')


# def plot_layer_scatters(net, figsize=(20, 10), title=None, save=None):
#     """
#     Plot weight and gradient norm history for each layer in layer_list across epochs
#     :param net: Either netG or netD Class
#     :param figsize: Desired size of figure
#     :param title: Title to be attached to save file. Recommended to differentiate between discriminator and generator
#     :param save: File path to save the resulting plot. If None, plot is not saved.
#     """
#     f, axes = plt.subplots(len(net.layer_list), 4, figsize=figsize, sharex=True)
#
#     axes[0, 0].title.set_text("Weight Norms")
#     axes[0, 1].title.set_text("Weight Gradient Norms")
#     axes[0, 2].title.set_text("Bias Norms")
#     axes[0, 3].title.set_text("Bias Gradient Norms")
#
#     for i in range(4):
#         axes[len(net.layer_list) - 1, i].set_xlabel('epochs')
#
#     for i, layer in enumerate(net.layer_list):
#         axes[i, 0].set_ylabel(net.layer_list_names[i])
#         axes[i, 0].plot(net.wnorm_history[layer]['weight'])
#         axes[i, 1].plot(net.gnorm_history[layer]['weight'])
#         axes[i, 2].plot(net.wnorm_history[layer]['bias'])
#         axes[i, 3].plot(net.gnorm_history[layer]['bias'])
#
#     if title:
#         sup = title + " Layer Weight and Gradient Norms"
#     else:
#         sup = "Layer Weight and Gradient Norms"
#     st = f.suptitle(sup, fontsize='x-large')
#     f.tight_layout()
#     st.set_y(0.96)
#     f.subplots_adjust(top=0.9)
#
#     f.show()
#
#     if save is not None:
#         assert os.path.exists(save), "Check that the desired save path exists."
#         safe_mkdir(save + '/layer_scatters')
#         f.savefig(save + '/layer_scatters/' + title + '_layer_scatter.png')


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


# def process_fake_output(raw_fake_output, le_dict):
#     """
#     :param raw_fake_output: Data generated by netG
#     :param le_dict: Dictionary of LabelEncoders to be used for inverse transformation back to original raw data
#     :return: Generated data inverse transformed and prepared for train_test_logistic_reg method. Data is still scaled and one hot encoded.
#     """
#     curr = 0
#     new_fake_output = np.copy(raw_fake_output)
#     for _, le in le_dict.items():
#         n = len(le.classes_)
#         newcurr = curr + n
#         max_idx = np.argmax(raw_fake_output[:, curr:newcurr], 1)
#         new_fake_output[:, curr:newcurr] = np.eye(n)[max_idx]
#         curr = newcurr
#     return new_fake_output


# def fully_process_fake_output(processed_fake_output, genned_labels, label_name, preprocessed_cat_mask, ohe, le_dict, scaler, cat_inputs, cont_inputs, int_inputs):
#     """
#     :param processed_fake_output: Output of process_fake_output method
#     :param genned_labels: Labels corresponding to generated data
#     :param label_name: Name of the dependent variable feature
#     :param preprocessed_cat_mask: Boolean mask of which features in array are categorical (True) versus continuous (False)
#     :param ohe: One Hot Encoder used to inverse transform data
#     :param le_dict: Dictionary of LabelEncoders to be used for inverse transformation back to original raw data
#     :param scaler: Scaler used to scale data
#     :param cat_inputs: List of names of categorical features in original raw data
#     :param cont_inputs: List of names of continuous features in original raw data
#     :param int_inputs: List of names of integer features in original raw data
#     :return: Generated data fully inverse transformed to be on the same basis as the original raw data
#     """
#     df = pd.DataFrame(index=range(processed_fake_output.shape[0]), columns=[label_name] + list(cat_inputs) + list(cont_inputs))
#
#     # Add labels
#     df[label_name] = genned_labels.astype('int')
#
#     # Split into cat and cont variables
#     cat_arr = processed_fake_output[:, preprocessed_cat_mask]
#     cont_arr = processed_fake_output[:, ~preprocessed_cat_mask]
#
#     # Inverse transform categorical variables
#     numerics = ohe.inverse_transform(cat_arr)
#     for i, le in enumerate(le_dict.items()):
#         df[le[0]] = le[1].inverse_transform(numerics[:, i].astype('int'))
#
#     # Inverse transform continuous variables
#     og_cont_arr = scaler.inverse_transform(cont_arr)
#     df[cont_inputs] = og_cont_arr
#
#     # Round integer inputs
#     df[int_inputs] = df[int_inputs].round(decimals=0).astype('int')
#
#     return df


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
