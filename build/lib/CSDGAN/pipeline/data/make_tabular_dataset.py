from utils.data_loading import *
from CSDGAN.utils.utils import *
from CSDGAN.classes.Tabular.TabularDataset import TabularDataset
from CSDGAN.utils.db import query_set_status
from CSDGAN.utils.utils import setup_run_logger
import logging


def make_tabular_dataset(run_id, username, title, dep_var, cont_inputs, int_inputs, test_size):
    """
    Requirements of data set is that it is contained in a flat file and the continuous vs. categorical vs. integer vs. dependent
    variables are specified. It should also be specified how to deal with missing data (stretch goal).
    """
    setup_run_logger(name='dataset_func', username=username, title=title)
    logger = logging.getLogger('dataset_func')

    try:
        run_id = str(run_id)
        query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Preprocessing data'])

        # Create directory for current run and place unzipped data set there
        run_dir = os.path.join(cs.RUN_FOLDER, username, title)
        assert os.path.exists(run_dir)

        # Perform various checks and load in data
        path = os.path.join(cs.UPLOAD_FOLDER, run_id)
        file = os.listdir(path)[0]
        assert os.path.splitext(file)[1] in {'.txt', '.csv', '.zip'}, "Path is not zip or flat file"
        if os.path.splitext(file)[1] == '.zip':
            logger.info('Tabular file contained in zip. Unzipping...')
            zip_ref = zipfile.ZipFile(path, 'r')
            zip_ref.extractall(run_dir)
            zip_ref.close()

            unzipped_path = os.path.join(run_dir, os.path.splitext(file)[0])
            assert os.path.exists(unzipped_path), \
                "Flat file in zip not named the same as zip file"
            unzipped_file = os.listdir(unzipped_path)[0]
            assert os.path.splitext(unzipped_file)[1] in {'.txt', '.csv'}, \
                "Flat file in zip should be .txt or .csv"
            data = pd.read_csv(os.path.join(run_dir, unzipped_file), header=0)
        else:
            logger.info('Tabular file not contained in zip.')
            data = pd.read_csv(os.path.join(path, file), header=0)

        # Initialize data set object
        dataset = TabularDataset(df=data,
                                 dep_var=dep_var,
                                 cont_inputs=cont_inputs,
                                 int_inputs=int_inputs,
                                 test_size=test_size)
        logger.info('TabularDataset successfully created. Pickling and exiting.')

        # Pickle relevant objects
        with open(os.path.join(run_dir, "dataset.pkl"), "wb") as f:
            pkl.dump(dataset, f)

    except Exception as e:
        query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Error'])
        logger.exception('Error: %s', e)
        raise Exception('Intentionally failing process after broadly catching an exception.')
