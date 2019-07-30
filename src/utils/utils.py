import src.utils.constants as cs
import os
import pandas as pd
import shutil
import logging


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in cs.ALLOWED_EXTENSIONS


def safe_mkdir(path):
    """Create a directory if there isn't one already"""
    try:
        os.mkdir(path)
    except OSError:
        pass


def new_run_mkdir(directory, username, title):
    """Initialize directories for a new run. Clears out prior uploads with same title."""
    safe_mkdir(os.path.join(directory, username))
    try:
        shutil.rmtree(os.path.join(directory, username, title))
    except OSError:
        pass
    safe_mkdir(os.path.join(directory, username, title))
    return os.path.join(directory, username, title)


def parse_tabular(directory, run_id):
    """Parses an uploaded tabular data set and returns a list of columns"""
    run_id = str(run_id)
    filename = os.listdir(os.path.join(directory, run_id))[0]
    data = pd.read_csv(os.path.join(directory, run_id, filename), nrows=0)
    return data.columns


def parse_dep(directory, run_id, dep_var):
    """Parses an uploaded tabular data set and returns a list of unique values for the dependent variable"""
    run_id = str(run_id)
    filename = os.listdir(os.path.join(directory, run_id))[0]
    data = pd.read_csv(os.path.join(directory, run_id, filename), usecols=[dep_var])
    return sorted(data[dep_var].unique())


def validate_tabular_choices(dep_var, cont_inputs, int_inputs):
    """Checks to see if user choices are consistent with expectations"""
    if dep_var in cont_inputs:
        return 'Dependent variable should not be continuous'
    if dep_var in int_inputs:
        return 'Dependent variable should not be integer'
    if any([int_input not in cont_inputs for int_input in int_inputs]):
        return 'Selected integer features should be a subset of selected continuous features'
    return None


def parse_image(upload_folder, username, title):
    # TODO
    pass


def setup_logger(name, username, title, filename='run_log', level=logging.INFO):
    log_setup = logging.getLogger(name)

    fileHandler = logging.FileHandler(os.path.join(cs.RUN_FOLDER, username, title, filename + '.log'), mode='a')
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler.setFormatter(formatter)

    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
