import CSDGAN.utils.constants as cs
import CSDGAN.utils.utils as cu
import CSDGAN.utils.db as db
import utils.utils as uu

import matplotlib.pyplot as plt
import numpy as np
import os


def build_img(img_key, username, title, run_id):
    """
    Generates a variety of different simple visualizations based on passed img_key
    :param img_key: Specifies which visualization to construct
    :param username: To locate model
    :param title: To locate model
    :param run_id: To locate model
    :return: Outputs a .png file to the viz folder
    """
    CGAN = cu.get_CGAN(username=username, title=title)
    viz_folder = os.path.join(cs.VIZ_FOLDER, username, title)
    os.makedirs(viz_folder, exist_ok=True)
    if img_key == cs.FILENAME_TRAINING_PLOT:
        CGAN.plot_training_plots(show=False, save=viz_folder)
    elif img_key == cs.FILENAME_PLOT_PROGRESS:
        import matplotlib; matplotlib.use('Agg')
        benchmark_acc = db.query_get_benchmark(run_id=run_id)
        CGAN.plot_progress(benchmark_acc=benchmark_acc, show=False, save=viz_folder)
    elif img_key == cs.FILENAME_netG_LAYER_SCATTERS:
        CGAN.netG.plot_layer_scatters(show=False, save=viz_folder)
    elif img_key == cs.FILENAME_netD_LAYER_SCATTERS:
        CGAN.netD.plot_layer_scatters(show=False, save=viz_folder)


def build_histograms(net, epoch, username, title):
    """
    Generates layer histograms based on specified epoch
    :param net: Which net to generate histograms of ("Discriminator" or "Generator")
    :param epoch: Which epoch to generate histograms of (INTEGER)
    :param username: To locate model
    :param title: To locate model
    :return: Outputs a .png file to the viz folder
    """
    CGAN = cu.get_CGAN(username=username, title=title)
    viz_folder = os.path.join(cs.VIZ_FOLDER, username, title)
    os.makedirs(viz_folder, exist_ok=True)
    epoch = int(epoch)
    if net == "Discriminator":
        CGAN.netD.plot_layer_hists(epoch=epoch, show=False, save=viz_folder)
    elif net == "Generator":
        CGAN.netG.plot_layer_hists(epoch=epoch, show=False, save=viz_folder)


def build_scatter_matrices(size, username, title, run_id):
    """
    Generates scatter matrices for a tabular CGAN with a specified data set size
    :param size: Size of data set to generate
    :param username: To locate model
    :param title: To locate model
    :param run_id: To locate model
    :return: Outputs a pair of .png files to the viz folder (real and fake)
    """
    size = int(size)
    CGAN = cu.get_CGAN(username=username, title=title)
    viz_folder = os.path.join(cs.VIZ_FOLDER, username, title)
    os.makedirs(viz_folder, exist_ok=True)

    genned_df = CGAN.gen_data(size=size, stratify=None)
    cont_inputs = db.query_get_cont_inputs(run_id=run_id)

    real_df = CGAN.gen_og_data()

    uu.plot_scatter_matrix(df=genned_df, cont_inputs=cont_inputs, title=cs.AVAILABLE_TABULAR_VIZ['scatter_matrix']['fake_title'], show=False, save=viz_folder)
    uu.plot_scatter_matrix(df=real_df, cont_inputs=cont_inputs, title=cs.AVAILABLE_TABULAR_VIZ['scatter_matrix']['real_title'], show=False, save=viz_folder)


def build_compare_cats(size, x, hue, username, title):
    """
    Generates categorical feature comparisons for a tabular CGAN with a specified data set size, and 2 categorical feature columns.
    """
    size = int(size)
    CGAN = cu.get_CGAN(username=username, title=title)
    viz_folder = os.path.join(cs.VIZ_FOLDER, username, title)
    os.makedirs(viz_folder, exist_ok=True)

    genned_df = CGAN.gen_data(size=size, stratify=None)
    real_df = CGAN.gen_og_data()
    dep_var = CGAN.data_gen.dataset.dep_var

    uu.compare_cats(real_df=real_df, fake_df=genned_df, x=x, hue=hue, y=dep_var, show=False, save=viz_folder)


def build_conditional_scatter(size, col1, col2, username, title):
    """Generates a conditional scatter plot for a tabular CGAN with a specified data set size, and 2 continuous features"""
    size = int(size)
    CGAN = cu.get_CGAN(username=username, title=title)
    viz_folder = os.path.join(cs.VIZ_FOLDER, username, title)
    os.makedirs(viz_folder, exist_ok=True)

    genned_df = CGAN.gen_data(size=size, stratify=None)
    real_df = CGAN.gen_og_data()
    dep_var = CGAN.data_gen.dataset.dep_var
    cont_inputs = CGAN.data_gen.dataset.cont_inputs
    labels_list = CGAN.data_gen.dataset.labels_list
    scaler = None  # Already handled when generating data

    uu.plot_conditional_scatter(real_df=real_df, fake_df=genned_df, col1=col1, col2=col2, dep_var=dep_var,
                                cont_inputs=cont_inputs, labels_list=labels_list, scaler=scaler, alpha=0.25,
                                show=False, save=viz_folder)


def build_conditional_density(size, col, username, title):
    """Generates a conditional scatter plot for a tabular CGAN with a specified data set size, and 2 continuous features"""
    size = int(size)
    CGAN = cu.get_CGAN(username=username, title=title)
    viz_folder = os.path.join(cs.VIZ_FOLDER, username, title)
    os.makedirs(viz_folder, exist_ok=True)

    genned_df = CGAN.gen_data(size=size, stratify=None)
    real_df = CGAN.gen_og_data()
    dep_var = CGAN.data_gen.dataset.dep_var
    cont_inputs = CGAN.data_gen.dataset.cont_inputs
    labels_list = CGAN.data_gen.dataset.labels_list
    scaler = None  # Already handled when generating data

    uu.plot_conditional_density(real_df=real_df, fake_df=genned_df, col=col, dep_var=dep_var,
                                cont_inputs=cont_inputs, labels_list=labels_list, scaler=scaler,
                                show=False, save=viz_folder)


def build_img_grid(labels, num_examples, epoch, username, title):
    """Generates an image of grids for an image CGAN with a specified epoch, labels, and number of examples of each label"""
    epoch, num_examples = int(epoch), int(num_examples)
    CGAN = cu.get_CGAN(username=username, title=title)
    viz_folder = os.path.join(cs.VIZ_FOLDER, username, title, 'imgs')
    os.makedirs(viz_folder, exist_ok=True)

    grid = CGAN.get_grid(index=epoch, labels=labels, num_examples=num_examples)
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.suptitle('Epoch ' + str(epoch))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    os.makedirs(viz_folder, exist_ok=True)
    img_name = os.path.join(viz_folder, 'Epoch ' + str(epoch) + '.png')
    plt.savefig(img_name)


def build_img_gif(labels, num_examples, start, stop, freq, fps, final_img_frames, username, title):
    """Generates a gif of images describing the effects of training over time with a specified epoch, labels, and number of examples of each label"""
    num_examples, start, stop, freq, fps, final_img_frames = int(num_examples), int(start), int(stop), int(freq), int(fps), int(final_img_frames)
    CGAN = cu.get_CGAN(username=username, title=title)
    viz_folder = os.path.join(cs.VIZ_FOLDER, username, title)
    os.makedirs(viz_folder, exist_ok=True)

    CGAN.build_gif(labels=labels, num_examples=num_examples, start=start, stop=stop, freq=freq, fps=fps, final_img_frames=final_img_frames, path=viz_folder)
