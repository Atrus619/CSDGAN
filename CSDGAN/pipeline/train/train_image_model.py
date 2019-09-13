import CSDGAN.utils.constants as cs
import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu
from CSDGAN.classes.image.ImageCGAN import ImageCGAN

import logging
import os
import torch
import pickle as pkl


def train_image_model(run_id, username, title, num_epochs, bs, nc, num_channels):
    """
    Trains an Image CGAN on the data preprocessed by make_image_dataset.py. Loads best generator and pickles CGAN for predictions.
    """
    run_id = str(run_id)
    db.query_verify_live_run(run_id=run_id)

    cu.setup_run_logger(name='train_func', username=username, title=title)
    cu.setup_run_logger(name='train_info', username=username, title=title, filename='train_log')
    logger = logging.getLogger('train_func')

    try:
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Train 0/4'])

        # Check for objects created by make_image_dataset.py
        run_dir = os.path.join(cs.RUN_FOLDER, username, title)
        exp_obj_list = ['le.pkl', 'ohe.pkl', 'train_gen.pkl', 'val_gen.pkl', 'test_gen.pkl']
        for exp_obj in exp_obj_list:
            assert os.path.exists(os.path.join(run_dir, exp_obj)), exp_obj + ' object not found'

        # Load prior constructed objects and instantiate CGAN object
        with open(os.path.join(run_dir, "le.pkl"), "rb") as f:
            le = pkl.load(f)

        with open(os.path.join(run_dir, "ohe.pkl"), "rb") as f:
            ohe = pkl.load(f)

        with open(os.path.join(run_dir, "train_gen.pkl"), "rb") as f:
            train_gen = pkl.load(f)

        with open(os.path.join(run_dir, "val_gen.pkl"), "rb") as f:
            val_gen = pkl.load(f)

        with open(os.path.join(run_dir, "test_gen.pkl"), "rb") as f:
            test_gen = pkl.load(f)

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        CGAN = ImageCGAN(train_gen=train_gen,
                         val_gen=val_gen,
                         test_gen=test_gen,
                         device=device,
                         nc=nc,
                         num_channels=num_channels,
                         path=run_dir,
                         le=le,
                         ohe=ohe,
                         fake_bs=bs,
                         **cs.IMAGE_CGAN_INIT_PARAMS)

        logger.info('Successfully instantiated CGAN object. Beginning training...')

        # Train TODO: Runtime Error catching issue with CUDA?
        CGAN.train_gan(num_epochs=num_epochs,
                       print_freq=cs.IMAGE_DEFAULT_PRINT_FREQ,
                       eval_freq=cs.IMAGE_DEFAULT_EVAL_FREQ,
                       run_id=run_id,
                       logger=logging.getLogger('train_info'))

        logger = logging.getLogger('train_func')
        logger.info('Successfully trained CGAN. Loading and saving best model...')

        # Load best-performing GAN and pickle CGAN to main directory
        CGAN.load_netG(best=True)

        with open(os.path.join(run_dir, 'CGAN.pkl'), 'wb') as f:
            pkl.dump(CGAN, f)

        logger.info('Successfully completed train_tabular_model function.')

    except Exception as e:
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Error'])
        logger.exception('Error: %s', e)
        raise Exception("Intentionally failing process after broadly catching an exception. "
                        "Logs describing this error can be found in the run's specific logs file.")
