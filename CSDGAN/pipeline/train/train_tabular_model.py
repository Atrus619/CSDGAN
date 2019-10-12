import CSDGAN.utils.constants as cs
import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu
import utils.utils as uu
from CSDGAN.classes.tabular.TabularCGAN import TabularCGAN

import logging
import os
import torch
import pickle as pkl
from torch.utils import data


def train_tabular_model(run_id, username, title, num_epochs, bs, tabular_init_params, tabular_eval_params, tabular_eval_folds):
    """
    Trains a Tabular CGAN on the data preprocessed by make_tabular_dataset.py. Loads best generator and pickles CGAN for predictions.
    """
    run_id = str(run_id)
    db.query_verify_live_run(run_id=run_id)

    cu.setup_run_logger(name='train_func', username=username, title=title)
    cu.setup_run_logger(name='train_info', username=username, title=title, filename='train_log')
    logger = logging.getLogger('train_func')

    try:
        # Check for objects created by make_tabular_dataset.py
        run_dir = os.path.join(cs.RUN_FOLDER, username, title)
        dataset = cu.get_tabular_dataset(username=username, title=title)

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        if len(pkl.dumps(dataset, -1)) < cs.TABULAR_MEM_THRESHOLD:
            dataset.to_dev(device)

        data_gen = data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)

        CGAN = TabularCGAN(data_gen=data_gen,
                           device=device,
                           path=run_dir,
                           seed=None,
                           eval_param_grid=tabular_eval_params,
                           eval_folds=tabular_eval_folds,
                           test_ranges=[dataset.x_train.shape[0] * 2 ** x for x in range(5)],
                           eval_stratify=dataset.eval_stratify,
                           nc=len(dataset.labels_list),
                           **tabular_init_params)

        # Benchmark and store
        logger.info('Successfully instantiated CGAN object. Beginning benchmarking...')
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Benchmarking'])
        benchmark = uu.train_test_logistic_reg(x_train=CGAN.data_gen.dataset.x_train.cpu().detach().numpy(),
                                               y_train=CGAN.data_gen.dataset.y_train.cpu().detach().numpy(),
                                               x_test=CGAN.data_gen.dataset.x_test.cpu().detach().numpy(),
                                               y_test=CGAN.data_gen.dataset.y_test.cpu().detach().numpy(),
                                               param_grid=tabular_eval_params,
                                               cv=tabular_eval_folds,
                                               labels_list=dataset.labels_list,
                                               verbose=False)
        db.query_update_benchmark(run_id=run_id, benchmark=benchmark)

        # Train
        logger.info('Successfully completed benchmark. Beginning training...')
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Train 0/4'])
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
        raise Exception("Intentionally failing process after broadly catching an exception. "
                        "Logs describing this error can be found in the run's specific logs file.")
