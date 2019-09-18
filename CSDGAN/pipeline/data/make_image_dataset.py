import CSDGAN.utils.constants as cs
import CSDGAN.utils.db as db
import CSDGAN.utils.utils as cu
import CSDGAN.utils.img_data_loading as cuidl

import logging
import os
import pickle as pkl


def make_image_dataset(run_id, username, title, folder, bs, x_dim=None, splits=None):  # TODO: Set default for splits/bs/x_dim
    """
    Requirements of image data set is that it should be a single zip with all images with same label in a folder named with the label name
    Images should either be the same size, or a specified image size should be provided (all images will be cropped to the same size)
    Assumes that file has been pre-unzipped and checked by the create.py functions/related util functions
    This file accomplishes the following:
        1. Accepts a desired image size (optional, else first image dim will be used), batch size, and train/val/test splits
        2. Splits data into train/val/test splits via stratified sampling and moves into corresponding folders
        3. Deletes original unzipped images
        4. Pickles label encoder, one hot encoder, resulting image size, and all three generators
    """
    run_id = str(run_id)
    db.query_verify_live_run(run_id=run_id)

    cu.setup_run_logger(name='dataset_func', username=username, title=title)
    logger = logging.getLogger('dataset_func')

    try:
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Preprocessing data'])

        # Check existence of run directory
        run_dir = os.path.join(cs.RUN_FOLDER, username, title)
        assert os.path.exists(run_dir), "Run directory does not exist"

        unzipped_path = os.path.join(run_dir, folder)
        assert os.path.exists(unzipped_path), "Unzipped path does not exist"

        # Load and preprocess data
        import_gen = cuidl.import_dataset(path=unzipped_path, bs=bs, shuffle=False, incl_paths=True)

        splits = [float(num) for num in splits]
        le, ohe, x_dim = cuidl.preprocess_imported_dataset(path=unzipped_path, import_gen=import_gen,
                                                           splits=splits, x_dim=x_dim)

        logger.info('Data successfully imported and preprocessed. Splitting into train/val/test...')

        # Create data loader for each component of data set
        train_gen = cuidl.import_dataset(os.path.join(unzipped_path, 'train'), bs=bs, shuffle=True, incl_paths=False)
        val_gen = cuidl.import_dataset(os.path.join(unzipped_path, 'val'), bs=bs, shuffle=False, incl_paths=False)
        test_gen = cuidl.import_dataset(os.path.join(unzipped_path, 'test'), bs=bs, shuffle=False, incl_paths=False)

        logger.info('Data successfully split into train/va/test. Pickling and exiting.')

        # Pickle relevant objects
        with open(os.path.join(run_dir, "le.pkl"), "wb") as f:
            pkl.dump(le, f)

        with open(os.path.join(run_dir, "ohe.pkl"), "wb") as f:
            pkl.dump(ohe, f)

        with open(os.path.join(run_dir, "train_gen.pkl"), "wb") as f:
            pkl.dump(train_gen, f)

        with open(os.path.join(run_dir, "val_gen.pkl"), "wb") as f:
            pkl.dump(val_gen, f)

        with open(os.path.join(run_dir, "test_gen.pkl"), "wb") as f:
            pkl.dump(test_gen, f)

    except Exception as e:
        db.query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Error'])
        logger.exception('Error: %s', e)
        raise Exception("Intentionally failing process after broadly catching an exception. "
                        "Logs describing this error can be found in the run's specific logs file.")
