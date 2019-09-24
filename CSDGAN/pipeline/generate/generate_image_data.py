import CSDGAN.utils.constants as cs
import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu

import logging
import os
import pickle as pkl
import numpy as np
import shutil


def generate_image_data(run_id, username, title, aug=None):
    """
    Loads an Image CGAN created by train_image_model.py. Generates data based on user specifications in pre-built gen_dict.pkl.
    :param aug: Whether this is part of the standard run or generating additional data
    """
    if aug is None:
        run_id = str(run_id)
        db.query_verify_live_run(run_id=run_id)

        cu.setup_run_logger(name='gen_func', username=username, title=title)
        logger = logging.getLogger('gen_func')

    try:
        if aug is None:
            db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Generating data'])

        # Check for objects created by train_image_model.py
        run_dir = os.path.join(cs.RUN_FOLDER, username, title)
        assert os.path.exists(os.path.join(run_dir, 'CGAN.pkl')), "CGAN object not found"
        if aug:
            gen_dict_path = os.path.join(run_dir, cs.GEN_DICT_NAME + ' Additional Data ' + str(aug) + '.pkl')
        else:
            gen_dict_path = os.path.join(run_dir, cs.GEN_DICT_NAME + '.pkl')

        assert os.path.exists(gen_dict_path), "gen_dict object not found"

        # Load in CGAN and gen_dict
        CGAN = cu.get_CGAN(username=username, title=title)

        with open(gen_dict_path, 'rb') as f:
            gen_dict = pkl.load(f)

        if aug is None:
            logger.info('Successfully loaded in CGAN. Generating data...')

        # Generate and output data
        folder_name = title + ('' if aug is None else ' Additional Data ' + str(aug))
        output_path = os.path.join(cs.OUTPUT_FOLDER, username, folder_name)
        os.makedirs(output_path, exist_ok=True)

        for i, (dep_class, size) in enumerate(gen_dict.items()):
            if size > 0:
                class_path = os.path.join(output_path, dep_class)
                os.makedirs(class_path, exist_ok=True)
                stratify = np.eye(CGAN.nc)[i]
                CGAN.gen_data(size=size, path=class_path, stratify=stratify, label=dep_class)

        _ = shutil.make_archive(output_path, 'zip', output_path)

        if aug is None:
            db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Complete'])
            logger.info('Successfully completed generate_tabular_data function. Run complete.')

    except Exception as e:
        if aug is None:
            db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Error'])
            logger.exception('Error: %s', e)
        raise Exception("Intentionally failing process after broadly catching an exception. "
                        "Logs describing this error can be found in the run's specific logs file.")
