import os
import CSDGAN.utils.constants as cs
import CSDGAN.utils.utils as cu
import CSDGAN.utils.db as db


def build_img(img_key, username, title, run_id):
    CGAN = cu.get_CGAN(username=username, title=title)
    viz_folder = os.path.join(cs.VIZ_FOLDER, username, title)
    os.makedirs(viz_folder, exist_ok=True)
    if img_key == cs.FILENAME_TRAINING_PLOT:
        CGAN.plot_training_plots(show=False, save=viz_folder)
    elif img_key == cs.FILENAME_PLOT_PROGRESS:
        benchmark_acc = db.query_get_benchmark(run_id=run_id)
        CGAN.plot_progress(benchmark_acc=benchmark_acc, show=False, save=viz_folder)

