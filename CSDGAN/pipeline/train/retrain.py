import CSDGAN.utils.constants as cs
import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu

import logging
import os
import pickle as pkl


def retrain(run_id, username, title, num_epochs):
    """
    Continues training a tabular model for a specified number of epochs
    """
    run_id = str(run_id)
    db.query_verify_live_run(run_id=run_id)

    cu.setup_run_logger(name='train_func', username=username, title=title)
    cu.setup_run_logger(name='train_info', username=username, title=title, filename='train_log')
    logger = logging.getLogger('train_func')

    try:
        run_dir = os.path.join(cs.RUN_FOLDER, username, title)

        # Load in prior trained GAN
        CGAN = cu.get_CGAN(username=username, title=title)

        # Train
        logger.info('Beginning retraining...')
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Retrain 0/4'])

        if type(CGAN).__name__ == 'TabularCGAN':
            CGAN.train_gan(num_epochs=num_epochs,
                           cadence=cs.TABULAR_DEFAULT_CADENCE,
                           print_freq=cs.TABULAR_DEFAULT_PRINT_FREQ,
                           eval_freq=cs.TABULAR_DEFAULT_EVAL_FREQ,
                           run_id=run_id,
                           logger=logging.getLogger('train_info'))
        elif type(CGAN).__name__ == 'ImageCGAN':
            CGAN.train_gan(num_epochs=num_epochs,
                           print_freq=cs.IMAGE_DEFAULT_PRINT_FREQ,
                           eval_freq=cs.IMAGE_DEFAULT_EVAL_FREQ,
                           run_id=run_id,
                           logger=logging.getLogger('train_info'))
        else:
            raise Exception('Invalid CGAN class object loaded')

        logger = logging.getLogger('train_func')
        logger.info('Successfully retrained CGAN. Loading and saving best model...')

        # Load best-performing GAN and pickle CGAN to main directory
        CGAN.load_netG(best=True)

        with open(os.path.join(run_dir, 'CGAN.pkl'), 'wb') as f:
            pkl.dump(CGAN, f)

        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Retraining Complete'])
        logger.info('Successfully completed retrain_tabular_model function.')

    except Exception as e:
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Error'])
        logger.exception('Error: %s', e)
        raise Exception("Intentionally failing process after broadly catching an exception. "
                        "Logs describing this error can be found in the run's specific logs file.")