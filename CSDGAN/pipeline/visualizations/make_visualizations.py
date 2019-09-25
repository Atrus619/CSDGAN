import os
import CSDGAN.utils.constants as cs
import CSDGAN.utils.utils as cu


def build_img(img_key, username, title):
    CGAN = cu.get_CGAN(username=username, title=title)
    viz_folder = os.path.join(cs.VIZ_FOLDER, username, title)
    os.makedirs(viz_folder, exist_ok=True)
    if img_key == cs.FILENAME_TRAINING_PLOT:
        CGAN.plot_training_plots(show=False, save=viz_folder)
    else:
        pass
