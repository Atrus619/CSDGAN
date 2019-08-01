from utils.data_loading import *
import src.utils.constants as cs
from utils.db import query_set_status
from src.utils.utils import setup_run_logger, export_tabular_to_zip
import logging


def generate_tabular_data(run_id, username, title):
    """
    Loads a tabular CGAN created by train_tabular_model.py. Generates data based on user specifications in pre-built gen_dict.pkl.
    """
    setup_run_logger(name='gen_func', username=username, title=title)
    logger = logging.getLogger('gen_func')

    try:
        run_id = str(run_id)
        query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Generating data'])

        # Check for objects created by train_tabular_model.py
        run_dir = os.path.join(cs.RUN_FOLDER, username, title)
        assert os.path.exists(os.path.join(run_dir, 'CGAN.pkl')), \
            "CGAN object not found"
        assert os.path.exists(os.path.join(run_dir, cs.TABULAR_GEN_DICT_NAME)), \
            "gen_dict object not found"

        # Load in CGAN and gen_dict
        with open(os.path.join(run_dir, 'CGAN.pkl'), 'rb') as f:
            CGAN = pkl.load(f)

        with open(os.path.join(run_dir, cs.TABULAR_GEN_DICT_NAME), 'rb') as f:
            gen_dict = pkl.load(f)

        logger.info('Successfully loaded in CGAN. Generating data...')

        # Generate data
        df = pd.DataFrame(columns=CGAN.data_gen.dataset.df_cols)
        for i, (dep_class, size) in enumerate(gen_dict.items()):
            if size > 0:
                stratify = np.eye(CGAN.nc)[i]
                tmp_df = CGAN.gen_data(size=size, stratify=stratify)
                tmp_df = tmp_df[df.columns.to_list()]
                df = pd.concat((df, tmp_df), axis=0)

        logger.info('Successfully generated data. Saving output to file...')

        # Output data
        export_tabular_to_zip(df=df, username=username, title=title)

        query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Complete'])

        logger.info('Successfully completed generate_tabular_data function. Run complete.')

    except Exception as e:
        query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Error'])
        logger.exception('Error: %s', e)
        raise Exception('Intentionally failing process after broadly catching an exception.')
