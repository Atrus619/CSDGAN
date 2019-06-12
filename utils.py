import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns


# Helper function to repeatedly test and print outputs for a logistic regression
def train_test_logistic_reg(x_train, y_train, x_test, y_test, param_grid, cv=5, random_state=None, labels=None):
    lr = LogisticRegression(penalty='elasticnet', multi_class='multinomial', solver='saga', random_state=random_state, max_iter=10000)
    lr_cv = GridSearchCV(lr, param_grid=param_grid, n_jobs=-1, cv=cv)
    lr_cv.fit(x_train, y_train)
    best_score = lr_cv.score(x_test, y_test)
    print("Accuracy:", best_score)
    print("Best Parameters:", lr_cv.best_params_)
    predictions = lr_cv.predict(x_test)
    print(classification_report(y_test, predictions, labels=labels))
    print(confusion_matrix(y_test, predictions, labels=labels))
    return [lr_cv, best_score]


# Helper function to generate a labeled tensor for the CGAN based on an equal distribution of classes
def gen_labels(size, num_classes, labels_list):
    assert size // num_classes == size / num_classes, "Make sure size is divisible by num_classes"
    output_one_hot = np.empty((0, num_classes))
    one_hot = pd.get_dummies(labels_list)
    output_labels = np.empty(0)
    for i in range(num_classes):
        tmp_one_hot = np.empty((size // num_classes, num_classes))
        tmp_labels = np.full((size // num_classes), labels_list[i])
        output_labels = np.concatenate((output_labels, tmp_labels), axis=0)
        for j in range(num_classes):
            tmp_one_hot[:, j] = one_hot.iloc[i, j]
        output_one_hot = np.concatenate((output_one_hot, tmp_one_hot), axis=0)
        output_one_hot = torch.tensor(output_one_hot, dtype=torch.float)
    return [output_one_hot, output_labels]


# Helper to plot sepal length vs width
def plot_scatters(X, y, title, scaler=None):
    if scaler:
        X = scaler.inverse_transform(X)
    inv = pd.DataFrame(X).rename(columns={0: 'sepal_len', 1: 'sepal_wid', 2: 'petal_len', 3: 'petal_wid'})
    inv = pd.concat((inv, pd.DataFrame(y)), axis=1).rename(columns={0: 'species'})

    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.scatter(inv.sepal_len[inv.species == 'Iris-setosa'], inv.sepal_wid[inv.species == 'Iris-setosa'], c='r', label='setosa')
    plt.scatter(inv.sepal_len[inv.species == 'Iris-versicolor'], inv.sepal_wid[inv.species == 'Iris-versicolor'], c='b',
                label='versicolor')
    plt.scatter(inv.sepal_len[inv.species == 'Iris-virginica'], inv.sepal_wid[inv.species == 'Iris-virginica'], c='g',
                label='virginica')
    # plt.legend()
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title('Sepal Length vs Width')

    plt.subplot(1, 2, 2)
    plt.scatter(inv.petal_len[inv.species == 'Iris-setosa'], inv.petal_wid[inv.species == 'Iris-setosa'], c='r', label='setosa')
    plt.scatter(inv.petal_len[inv.species == 'Iris-versicolor'], inv.petal_wid[inv.species == 'Iris-versicolor'], c='b', label='versicolor')
    plt.scatter(inv.petal_len[inv.species == 'Iris-virginica'], inv.petal_wid[inv.species == 'Iris-virginica'], c='g', label='virginica')

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


# Helper to plot distributions of variables by class
def plot_densities(X, y, title, scaler=None):
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

    plt.show()


def training_plots(netD, netG, num_epochs):
    f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True)

    axes[0, 0].title.set_text("Generator and Discriminator Loss During Training")
    axes[0, 0].plot(netG.losses, label="G")
    axes[0, 0].plot(netD.losses, label="D")
    axes[0, 0].set_xlabel("iterations")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].title.set_text("Average Discriminator Outputs During Training")
    axes[0, 1].plot(netD.Avg_D_reals, label="R")
    axes[0, 1].plot(netD.Avg_D_fakes, label="F")
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


def fake_data_training_plots(real_range, score_real, test_range, fake_scores):
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
