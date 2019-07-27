from utils.data_loading import *
from src.utils import *
from classes.Tabular.TabularDataset import TabularDataset
from src.db import query_set_status


def make_tabular_dataset(run_id, username, title, dep_var, cont_inputs, int_inputs, test_size):
    """
    Requirements of data set is that it is contained in a flat file and the continuous vs. categorical vs. integer vs. dependent
    variables are specified. It should also be specified how to deal with missing data (stretch goal).
    """
    # Update status
    query_set_status(run_id=run_id, status_id=cs.STATUS_DICT['Preprocessing data'])

    # Create directory for current run and place unzipped data set there
    run_dir = os.path.join(cs.RUN_FOLDER, username, title)
    new_run_mkdir(directory=cs.RUN_FOLDER, username=username, title=title)

    # Perform various checks and load in data
    path = os.path.join(cs.UPLOAD_FOLDER, run_id)
    assert os.path.splitext(path)[1] in {'.txt', '.csv', '.zip'}, "Path is not zip or flat file"
    if os.path.splitext(path)[1] == '.zip':
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(run_dir)
        zip_ref.close()

        unzipped_path = os.path.join(run_dir, os.path.splitext(os.path.basename(path))[0])
        assert os.path.exists(unzipped_path), \
            "Flat file in zip not named the same as zip file"
        assert os.path.splitext(run_dir)[1] in {'.txt', '.csv'}, \
            "Flat file in zip should be .txt or .csv"
        data = pd.read_csv(run_dir, header=0)
    else:
        data = pd.read_csv(path, header=0)

    # Initialize data set object
    dataset = TabularDataset(df=data,
                             dep_var=dep_var,
                             cont_inputs=cont_inputs,
                             int_inputs=int_inputs,
                             test_size=test_size)

    # Pickle relevant objects
    with open(os.path.join(run_dir, "dataset.pkl"), "wb") as f:
        pkl.dump(dataset, f)
