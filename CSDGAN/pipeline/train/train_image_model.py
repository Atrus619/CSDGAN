import CSDGAN.utils.constants as cs
import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu
from CSDGAN.classes.image.ImageCGAN import ImageCGAN

import logging
import os
import torch
import pickle as pkl


def train_image_model(run_id, username, title, num_epochs, bs, nc, num_channels, image_init_params, image_eval_freq):
    """
    Trains an Image CGAN on the data preprocessed by make_image_dataset.py. Loads best generator and pickles CGAN for predictions.
    """
    run_id = str(run_id)
    db.query_verify_live_run(run_id=run_id)

    cu.setup_run_logger(name='train_func', username=username, title=title)
    cu.setup_run_logger(name='train_info', username=username, title=title, filename='train_log')
    logger = logging.getLogger('train_func')

    try:
        # Check for objects created by make_image_dataset.py
        run_dir = os.path.join(cs.RUN_FOLDER, username, title)
        le, ohe, train_gen, val_gen, test_gen = cu.get_image_dataset(username=username, title=title)

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
                         **image_init_params)

        # Benchmark and store
        logger.info('Successfully instantiated CGAN object. Beginning benchmarking...')
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Benchmarking'])

        benchmark, real_netE = CGAN.eval_on_real_data(num_epochs=image_init_params['eval_num_epochs'],
                                                      es=image_init_params['early_stopping_patience'])

        db.query_update_benchmark(run_id=run_id, benchmark=benchmark)

        with open(os.path.join(run_dir, 'real_netE.pkl'), 'wb') as f:
            pkl.dump(real_netE, f)

        # Train
        logger.info('Successfully completed benchmark. Beginning training...')
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Train 0/4'])
        CGAN.train_gan(num_epochs=num_epochs,
                       print_freq=cs.IMAGE_DEFAULT_PRINT_FREQ,
                       eval_freq=image_eval_freq,
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
