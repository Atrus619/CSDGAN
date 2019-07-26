import src.constants as cs
import os
import pandas as pd
import shutil


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in cs.ALLOWED_EXTENSIONS


def safe_mkdir(path):
    """Create a directory if there isn't one already"""
    try:
        os.mkdir(path)
    except OSError:
        pass


def new_run_mkdir(upload_folder, username, title):
    """Initialize directories for a new run. Clears out prior uploads with same title."""
    safe_mkdir(os.path.join(upload_folder, username))
    try:
        shutil.rmtree(os.path.join(upload_folder, username, title))
    except OSError:
        pass
    safe_mkdir(os.path.join(upload_folder, username, title))


def parse_tabular(upload_folder, username, title):
    """Parses an uploaded tabular data set and returns a list of columns"""
    filename = os.listdir(os.path.join(upload_folder, username, title))[0]
    data = pd.read_csv(os.path.join(upload_folder, username, title, filename), nrows=0)
    return data.columns


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
