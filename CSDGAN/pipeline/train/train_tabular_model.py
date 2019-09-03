import CSDGAN.utils.constants as cs
import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu
from CSDGAN.classes.tabular.TabularCGAN import TabularCGAN

import logging
import os
import torch
import pickle as pkl
from torch.utils import data


def train_tabular_model(run_id, username, title, num_epochs, bs):
    """
    Trains a Tabular CGAN on the data preprocessed by make_tabular_dataset.py. Loads best generator and pickles CGAN for predictions
    """
    cu.setup_run_logger(name='train_func', username=username, title=title)
    cu.setup_run_logger(name='train_info', username=username, title=title, filename='train_log')
    logger = logging.getLogger('train_func')

    try:
        run_id = str(run_id)
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Train 0/4'])

        # Check for objects created by make_tabular_dataset.py
        run_dir = os.path.join(cs.RUN_FOLDER, username, title)
        assert os.path.exists(os.path.join(run_dir, 'dataset.pkl')), \
            "Data set object not found"

        # Load data set and create CGAN object
        with open(os.path.join(run_dir, "dataset.pkl"), 'rb') as f:
            dataset = pkl.load(f)

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        if len(pkl.dumps(dataset, -1)) < cs.TABULAR_MEM_THRESHOLD:
            dataset.to_dev(device)

        data_gen = data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)

        CGAN = TabularCGAN(data_gen=data_gen,
                           device=device,
                           path=run_dir,
                           seed=None,
                           eval_param_grid=cs.TABULAR_EVAL_PARAM_GRID,
                           eval_folds=cs.TABULAR_EVAL_FOLDS,
                           test_ranges=[dataset.x_train.shape[0]*2**x for x in range(5)],
                           eval_stratify=dataset.eval_stratify,
                           nc=len(dataset.labels_list),
                           nz=cs.TABULAR_DEFAULT_NZ,
                           sched_netG=cs.TABULAR_DEFAULT_SCHED_NETG,
                           netG_H=cs.TABULAR_DEFAULT_NETG_H,
                           netD_H=cs.TABULAR_DEFAULT_NETD_H,
                           **cs.TABULAR_CGAN_INIT_PARAMS)

        logger.info('Successfully instantiated CGAN object. Beginning training...')

        # Train
        CGAN.train_gan(num_epochs=num_epochs,
                       cadence=cs.TABULAR_DEFAULT_CADENCE,
                       print_freq=cs.TABULAR_DEFAULT_PRINT_FREQ,
                       eval_freq=cs.TABULAR_DEFAULT_EVAL_FREQ,
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
        raise Exception('Intentionally failing process after broadly catching an exception. '
                        'Logs describing this error can be found in the run\'s specific logs file.')
