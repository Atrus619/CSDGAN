import os
import CSDGAN.utils.constants as cs
import CSDGAN.utils.utils as cu
import CSDGAN.utils.db as db
import utils.utils as uu


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
