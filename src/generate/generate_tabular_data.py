from utils.data_loading import *
import src.constants as cs
from src.db import query_set_status


def generate_tabular_data(run_id, username, title):
    """
    Loads a tabular CGAN created by train_tabular_model.py. Generates data based on user specifications in pre-built gen_dict.pkl.
    """
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

    # Generate data
    df = pd.DataFrame(columns=CGAN.data_gen.dataset.df_cols)
    for i, (dep_class, size) in enumerate(gen_dict.items()):
        stratify = np.eye(CGAN.nc)[i]
        tmp_df = CGAN.gen_data(size=size, stratify=stratify)
        tmp_df = tmp_df[df.columns.to_list()]
        df = pd.concat((df, tmp_df), axis=0)

    # Output data
    df.to_csv(cs.TABULAR_DEFAULT_GENNED_DATA_FILENAME, index=False)

    query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Complete'])
