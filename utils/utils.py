import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random


# Helper function to set working directory to the primary one
def fix_wd():
    while os.path.basename(os.getcwd()) != 'Synthetic_Data_GAN_Capstone':
        os.chdir('..')


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def gen_fake_data(netG, bs, nz, nc, labels_list, device, stratify=None):
    noise = torch.randn(bs, nz, device=device)
    fake_labels, output_labels = gen_labels(size=bs, num_classes=nc, labels_list=labels_list, stratify=stratify)
    fake_labels = fake_labels.to(device)
    fake_data = netG(noise, fake_labels).cpu().detach().numpy()
    return fake_data, output_labels


# Helper function to generate a labeled tensor for the CGAN based on an equal distribution of classes
def gen_labels(size, num_classes, labels_list, stratify=None):
    if stratify is None:
        assert size // num_classes == size / num_classes, "Make sure size is divisible by num_classes"
        stratify = np.full(num_classes, 1 / num_classes)
    else:
        assert np.sum(stratify) == 1, "Make sure stratify sums to 1"
    counts = np.round(np.dot(stratify, size), decimals=0).astype('int')
    while np.sum(counts) != size:
        if np.sum(counts) > size:
            counts[random.choice(range(num_classes))] -= 1
        else:
            counts[random.choice(range(num_classes))] += 1
    output_one_hot = np.empty((0, num_classes))
    one_hot = pd.get_dummies(labels_list)
    output_labels = np.empty(0)
    for i in range(num_classes):
        tmp_one_hot = np.empty((counts[i], num_classes))
        tmp_labels = np.full(counts[i], labels_list[i])
        output_labels = np.concatenate((output_labels, tmp_labels), axis=0)
        for j in range(num_classes):
            tmp_one_hot[:, j] = one_hot.iloc[i, j]
        output_one_hot = np.concatenate((output_one_hot, tmp_one_hot), axis=0)
        output_one_hot = torch.tensor(output_one_hot, dtype=torch.float)
    return [output_one_hot, output_labels]


# Train a model on fake data and evaluate on test data in order to evaluate network as it trains
def evaluate_training_progress(test_range, fake_bs, nz, nc, out_dim, netG, x_test, y_test, manualSeed, labels_list, param_grid, device, le_dict=None, stratify=None):
    fake_scores = []
    fake_models = []
    for size in test_range:
        genned_data = np.empty((0, out_dim))
        genned_labels = np.empty(0)
        rem = size
        while rem > 0:
            curr_size = min(fake_bs, rem)
            fake_data, output_labels = gen_fake_data(netG=netG, bs=curr_size, nz=nz, nc=nc, labels_list=labels_list, device=device, stratify=stratify)
            rem -= curr_size
            genned_data = np.concatenate((genned_data, fake_data))
            genned_labels = np.concatenate((genned_labels, output_labels))
        if le_dict is not None:
            genned_data = process_fake_output(genned_data, le_dict)
        model_fake_tmp, score_fake_tmp = train_test_logistic_reg(x_train=genned_data, y_train=genned_labels, x_test=x_test, y_test=y_test,
                                                                 param_grid=param_grid, cv=5, random_state=manualSeed, labels=labels_list, verbose=0)
        fake_models.append(model_fake_tmp)
        fake_scores.append(score_fake_tmp)
    return fake_models, fake_scores


# Helper function to repeatedly test and print outputs for a logistic regression
def train_test_logistic_reg(x_train, y_train, x_test, y_test, param_grid, cv=5, random_state=None, labels=None, verbose=1):
    lr = LogisticRegression(penalty='elasticnet', multi_class='multinomial', solver='saga', random_state=random_state, max_iter=10000)
    lr_cv = GridSearchCV(lr, param_grid=param_grid, n_jobs=-1, cv=cv, iid=True)
    lr_cv.fit(x_train, y_train)
    best_score = lr_cv.score(x_test, y_test)
    predictions = lr_cv.predict(x_test)
    if verbose == 1:
        print("Accuracy:", best_score)
        print("Best Parameters:", lr_cv.best_params_)
        print(classification_report(y_test, predictions, labels=labels))
        print(confusion_matrix(np.array(y_test), predictions, labels=labels))
    return [lr_cv, best_score]


# Plot progress so far on training
def plot_training_progress(stored_scores, test_range, num_saves, real_data_score, save=None):
    ys = np.empty((num_saves, len(test_range)))
    xs = np.empty((num_saves, len(test_range)))
    barWidth = 1 / (len(test_range) + 1)
    for i in range(len(test_range)):
        ys[:, i] = np.array(stored_scores[i:num_saves * len(test_range):len(test_range)])
        xs[:, i] = np.arange(num_saves) + barWidth * i
        plt.bar(xs[:, i], ys[:, i], width=barWidth, edgecolor='white', label=test_range[i])

    line_len = 2 * len(test_range) + 1
    plt.plot(np.linspace(0, line_len, line_len), np.full(line_len, real_data_score), linestyle='dashed', color='r')
    plt.xlabel('Epoch', fontweight='bold')
    plt.xticks([r + barWidth for r in range(num_saves)], list(range(num_saves)))
    plt.title('Evaluation Over Training Epochs')
    plt.legend(loc=4)
    plt.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/training_progress')
        plt.savefig(save + '/training_progress/' + 'training_progress.png')


# Helper/diagnostic function to return stats for a specific model
def parse_models(stored_models, epoch, print_interval, test_range, ind, x_test, y_test, labels):
    tmp_model = stored_models[epoch // print_interval * len(test_range) - 1 + ind]
    best_score = tmp_model.score(x_test, y_test)
    predictions = tmp_model.predict(x_test)
    print("Accuracy:", best_score)
    print("Best Parameters:", tmp_model.best_params_)
    print(classification_report(y_test, predictions, labels=labels))
    print(confusion_matrix(y_test, predictions, labels=labels))


# Plots scatter matrix of data set (real or fake)
def plot_scatter_matrix(X, title, og_df, scaler=None, save=None):
    if scaler:
        X = scaler.inverse_transform(X)
    pd.plotting.scatter_matrix(pd.DataFrame(X, columns=og_df.columns), figsize=(12, 12))
    plt.suptitle(title, fontsize='x-large')
    plt.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/scatter_matrices')
        plt.savefig(save + '/scatter_matrices/' + title + '_scatter_matrix.png')


# Plots a conditional scatter plot (labels are colors) to compare real and fake data side by side
def plot_conditional_scatter(x_real, y_real, x_fake, y_fake, col1, col2, class_dict, og_df, scaler=None, alpha=1.0, save=None):
    if scaler:
        x_real = scaler.inverse_transform(x_real)
        x_fake = scaler.inverse_transform(x_fake)

    f, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    axes[0].title.set_text("Real")
    axes[1].title.set_text("Fake")

    for label in class_dict:
        axes[0].scatter(x=x_real[:, col1][y_real == label], y=x_real[:, col2][y_real == label], label=class_dict[label][0], c=class_dict[label][1], alpha=alpha)
        axes[1].scatter(x=x_fake[:, col1][y_fake == label], y=x_fake[:, col2][y_fake == label], label=class_dict[label][0], c=class_dict[label][1], alpha=alpha)

    axes[0].set_xlabel(og_df.columns[col1])
    axes[1].set_xlabel(og_df.columns[col1])
    axes[0].set_ylabel(og_df.columns[col2])

    axes[1].legend()

    st = f.suptitle(og_df.columns[col1] + " vs. " + og_df.columns[col2], fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.85)

    f.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/conditional_scatters')
        f.savefig(save + '/conditional_scatters/' + og_df.columns[col1] + '_vs_' + og_df.columns[col2] + '_conditional_scatter.png')


# Plots a conditional density plot (labels are colors) to compare real and fake data side by side
def plot_conditional_density(x_real, y_real, x_fake, y_fake, col, class_dict, og_df, scaler=None, save=None):
    if scaler:
        x_real = scaler.inverse_transform(x_real)
        x_fake = scaler.inverse_transform(x_fake)

    f, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    axes[0].title.set_text("Real")
    axes[1].title.set_text("Fake")

    for label in class_dict:
        sns.distplot(a=x_real[:, col][y_real == label], label=class_dict[label][0], color=class_dict[label][1], ax=axes[0])
        sns.distplot(a=x_fake[:, col][y_fake == label], label=class_dict[label][0], color=class_dict[label][1], ax=axes[1])

    st = f.suptitle(og_df.columns[col] + ' conditional density plot', fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.85)

    f.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/conditional_densities')
        f.savefig(save + '/conditional_densities/' + og_df.columns[col] + '_conditional_density.png')


# Helper to plot iris sepal length vs width
def iris_plot_scatters(X, y, title, scaler=None, alpha=1.0, save=None):
    if scaler:
        X = scaler.inverse_transform(X)
    inv = pd.DataFrame(X).rename(columns={0: 'sepal_len', 1: 'sepal_wid', 2: 'petal_len', 3: 'petal_wid'})
    inv = pd.concat((inv, pd.DataFrame(y)), axis=1).rename(columns={0: 'species'})

    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.scatter(inv.sepal_len[inv.species == 'Iris-setosa'], inv.sepal_wid[inv.species == 'Iris-setosa'], c='r', label='setosa', alpha=alpha)
    plt.scatter(inv.sepal_len[inv.species == 'Iris-versicolor'], inv.sepal_wid[inv.species == 'Iris-versicolor'], c='b', label='versicolor', alpha=alpha)
    plt.scatter(inv.sepal_len[inv.species == 'Iris-virginica'], inv.sepal_wid[inv.species == 'Iris-virginica'], c='g', label='virginica', alpha=alpha)
    # plt.legend()
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title('Sepal Length vs Width')

    plt.subplot(1, 2, 2)
    plt.scatter(inv.petal_len[inv.species == 'Iris-setosa'], inv.petal_wid[inv.species == 'Iris-setosa'], c='r', label='setosa', alpha=alpha)
    plt.scatter(inv.petal_len[inv.species == 'Iris-versicolor'], inv.petal_wid[inv.species == 'Iris-versicolor'], c='b', label='versicolor', alpha=alpha)
    plt.scatter(inv.petal_len[inv.species == 'Iris-virginica'], inv.petal_wid[inv.species == 'Iris-virginica'], c='g', label='virginica', alpha=alpha)

    plt.legend()
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title('Petal Length vs Width')

    st = plt.suptitle(title + " Feature Scatter Plots", fontsize='x-large')
    plt.tight_layout()

    # shift subplots down:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    plt.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/conditional_scatters')
        plt.savefig(save + '/conditional_scatters/' + title + '_conditional_scatter.png')


# Helper to plot iris distributions of variables by class
def iris_plot_densities(X, y, title, scaler=None, save=None):
    if scaler:
        X = scaler.inverse_transform(X)
    inv = pd.DataFrame(X).rename(columns={0: 'sepal_len', 1: 'sepal_wid', 2: 'petal_len', 3: 'petal_wid'})
    inv = pd.concat((inv, pd.DataFrame(y)), axis=1).rename(columns={0: 'species'})

    f, axes = plt.subplots(3, 4, figsize=(8, 8), sharex=True, sharey=True)

    sns.distplot(inv['sepal_len'][inv.species == 'Iris-setosa'], color='red', ax=axes[0, 0])
    sns.distplot(inv['sepal_len'][inv.species == 'Iris-versicolor'], color='blue', ax=axes[1, 0])
    sns.distplot(inv['sepal_len'][inv.species == 'Iris-virginica'], color='green', ax=axes[2, 0])
    sns.distplot(inv['sepal_wid'][inv.species == 'Iris-setosa'], color='red', ax=axes[0, 1])
    sns.distplot(inv['sepal_wid'][inv.species == 'Iris-versicolor'], color='blue', ax=axes[1, 1])
    sns.distplot(inv['sepal_wid'][inv.species == 'Iris-virginica'], color='green', ax=axes[2, 1])
    sns.distplot(inv['petal_len'][inv.species == 'Iris-setosa'], color='red', ax=axes[0, 2])
    sns.distplot(inv['petal_len'][inv.species == 'Iris-versicolor'], color='blue', ax=axes[1, 2])
    sns.distplot(inv['petal_len'][inv.species == 'Iris-virginica'], color='green', ax=axes[2, 2])
    sns.distplot(inv['petal_wid'][inv.species == 'Iris-setosa'], color='red', ax=axes[0, 3])
    sns.distplot(inv['petal_wid'][inv.species == 'Iris-versicolor'], color='blue', ax=axes[1, 3])
    sns.distplot(inv['petal_wid'][inv.species == 'Iris-virginica'], color='green', ax=axes[2, 3])

    axes[0, 0].set_ylabel('Iris-setosa')
    axes[1, 0].set_ylabel('Iris-versicolor')
    axes[2, 0].set_ylabel('Iris-virginica')

    plt.xlim(0, 10)
    plt.ylim(0, 2)

    plt.xticks(list(range(11)))
    plt.yticks(list(np.linspace(0, 2, 11)))

    st = plt.suptitle(title + " Feature Density Plots", fontsize='x-large')
    plt.tight_layout()

    # shift subplots down:
    st.set_y(.95)
    f.subplots_adjust(top=0.9)

    f.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/conditional_densities')
        f.savefig(save + '/conditional_densities/' + title + '_conditional_density.png')


def training_plots(netD, netG, num_epochs, save=None):
    f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True)

    axes[0, 0].title.set_text("Generator and Discriminator Loss During Training")
    axes[0, 0].plot(netG.losses, label="G")
    axes[0, 0].plot(netD.losses, label="D")
    axes[0, 0].set_xlabel("iterations")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()

    axes[0, 1].title.set_text("Average Discriminator Outputs During Training")
    axes[0, 1].plot(netD.Avg_D_reals, label="Real")
    axes[0, 1].plot(netD.Avg_D_fakes, label="Fake")
    axes[0, 1].plot(np.linspace(0, num_epochs, num_epochs), np.full(num_epochs, 0.5))
    axes[0, 1].set_xlabel("iterations")
    axes[0, 1].set_ylabel("proportion")
    axes[0, 1].legend()

    axes[1, 0].title.set_text('Gradient Norm History')
    axes[1, 0].plot(netG.gnorm_total_hist, label="G")
    axes[1, 0].plot(netD.gnorm_total_hist, label="D")
    axes[1, 0].set_xlabel("iterations")
    axes[1, 0].set_ylabel("norm")
    axes[1, 0].legend()

    axes[1, 1].title.set_text('Weight Norm History')
    axes[1, 1].plot(netG.wnorm_total_hist, label="G")
    axes[1, 1].plot(netD.wnorm_total_hist, label="D")
    axes[1, 1].set_xlabel("iterations")
    axes[1, 1].set_ylabel("norm")
    axes[1, 1].legend()

    st = f.suptitle("Training Diagnostic Plots", fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)

    f.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/training_plots')
        f.savefig(save + '/training_plots/training_plot.png')


def fake_data_training_plots(real_range, score_real, test_range, fake_scores, save=None):
    f, axes = plt.subplots(1, 2, figsize=(8, 8))

    axes[0].title.set_text('Sample Sizes')
    labels = ['real' + str(real_range)] + ['fake' + str(x) for x in test_range]
    axes[0].bar(x=labels, height=[real_range] + test_range)
    axes[0].set_xticklabels(labels=labels, rotation='vertical')

    axes[1].title.set_text('OOF Accuracy')
    axes[1].bar(x=labels, height=[score_real] + fake_scores)
    axes[1].set_xticklabels(labels=labels, rotation='vertical')
    axes[1].set_ylim(0.8, 1.0)

    st = f.suptitle("Fake Data Training Results", fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)

    f.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/fake_data_training_plots')
        f.savefig(save + '/fake_data_training_plots/fake_data_training_plot.png')


def plot_layer_scatters(net, figsize=(20, 10), title=None, save=None):
    f, axes = plt.subplots(len(net.layer_list), 4, figsize=figsize, sharex=True)

    axes[0, 0].title.set_text("Weight Norms")
    axes[0, 1].title.set_text("Weight Gradient Norms")
    axes[0, 2].title.set_text("Bias Norms")
    axes[0, 3].title.set_text("Bias Gradient Norms")

    for i in range(4):
        axes[len(net.layer_list) - 1, i].set_xlabel('epochs')

    for i, layer in enumerate(net.layer_list):
        axes[i, 0].set_ylabel(net.layer_list_names[i])
        axes[i, 0].plot(net.wnorm_hist[layer]['weight'])
        axes[i, 1].plot(net.gnorm_hist[layer]['weight'])
        axes[i, 2].plot(net.wnorm_hist[layer]['bias'])
        axes[i, 3].plot(net.gnorm_hist[layer]['bias'])

    if title:
        sup = title + " Layer Weight and Gradient Norms"
    else:
        sup = "Layer Weight and Gradient Norms"
    st = f.suptitle(sup, fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)

    f.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/layer_scatters')
        f.savefig(save + '/layer_scatters/' + title + '_layer_scatter.png')


def scale_cont_inputs(arr, preprocessed_cat_mask, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        arr_cont = scaler.fit_transform(arr[:, ~preprocessed_cat_mask])
    else:
        arr_cont = scaler.transform(arr[:, ~preprocessed_cat_mask])
    arr_cat = arr[:, preprocessed_cat_mask]
    return np.concatenate((arr_cat, arr_cont), axis=1), scaler


def encode_categoricals_custom(df, x_train, x_test, cat_inputs, cat_mask):
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
    count = 0
    for _, le in le_dict.items():
        count += len(le.classes_)
    return np.concatenate([np.full(count, True), np.full(x_train.shape[1] - count, False)])


def process_fake_output(raw_fake_output, le_dict):
    curr = 0
    new_fake_output = np.copy(raw_fake_output)
    for _, le in le_dict.items():
        n = len(le.classes_)
        newcurr = curr + n
        max_idx = np.argmax(raw_fake_output[:, curr:newcurr], 1)
        new_fake_output[:, curr:newcurr] = np.eye(n)[max_idx]
        curr = newcurr
    return new_fake_output


def fully_process_fake_output(processed_fake_output, genned_labels, label_name, preprocessed_cat_mask, ohe, le_dict, scaler, cat_inputs, cont_inputs, int_inputs):
    df = pd.DataFrame(index=range(processed_fake_output.shape[0]), columns=[label_name] + list(cat_inputs) + list(cont_inputs))

    # Add labels
    df[label_name] = genned_labels.astype('int')

    # Split into cat and cont variables
    cat_arr = processed_fake_output[:, preprocessed_cat_mask]
    cont_arr = processed_fake_output[:, ~preprocessed_cat_mask]

    # Inverse transform categorical variables
    numerics = ohe.inverse_transform(cat_arr)
    for i, le in enumerate(le_dict.items()):
        df[le[0]] = le[1].inverse_transform(numerics[:, i].astype('int'))

    # Inverse transform continuous variables
    og_cont_arr = scaler.inverse_transform(cont_arr)
    df[cont_inputs] = og_cont_arr

    # Round integer inputs
    df[int_inputs] = df[int_inputs].round(decimals=0).astype('int')

    return df


def compare_cats(real, fake, x, y, hue, save=None):
    f, axes = plt.subplots(1, 2, figsize=(8, 8), sharey=True, sharex=True)

    axes[0].title.set_text('Fake Data')
    axes[1].title.set_text('Real Data')

    sns.catplot(data=real, x=x, y=y, hue=hue, kind='bar', ax=axes[0])
    sns.catplot(data=fake, x=x, y=y, hue=hue, kind='bar', ax=axes[1])

    sup = "Comparing {0} by {1} and {2}".format(y, x, hue)
    st = f.suptitle(sup, fontsize='x-large')
    f.tight_layout()
    st.set_y(0.96)
    f.subplots_adjust(top=0.9)

    # Quick n' dirty solution to sns.catplot creating its own blank subplots
    plt.close(2)
    plt.close(3)
    f.show()

    if save is not None:
        assert os.path.exists(save), "Check that the desired save path exists."
        safe_mkdir(save + '/compare_cats')
        f.savefig(save + '/compare_cats/' + x + '_' + hue + '_cat_comparison.png')


def train_val_test_split(x, y, splits, random_state=None):
    """
    Performs a train/validation/test split on x and y, stratified, based on desired splits
    :param x: independent var
    :param y: dependent var
    :param splits: proportion of total data to be assigned to train/validation/test set
    :param random_state: Optional random state
    :return:
    """
    assert sum(splits) == 1.0, "Please make sure sum of splits is equal to 1"
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=splits[2], stratify=y, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=splits[1] / splits[0], stratify=y_train, random_state=random_state)
    x_train, y_train, x_val, y_val, x_test, y_test = torch.from_numpy(x_train), torch.from_numpy(y_train), \
                                                     torch.from_numpy(x_val), torch.from_numpy(y_val), \
                                                     torch.from_numpy(x_test), torch.from_numpy(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test
