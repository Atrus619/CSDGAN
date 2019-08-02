import CSDGAN.utils.constants as cs
import os
import pandas as pd
import shutil
import logging
import unicodedata
import string
import datetime as d
from zipfile import ZipFile
import pickle as pkl


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


def setup_run_logger(name, username, title, filename='run_log', level=logging.INFO):
    log_setup = logging.getLogger(name)

    fileHandler = logging.FileHandler(os.path.join(cs.RUN_FOLDER, username, title, filename + '.log'), mode='a')
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler.setFormatter(formatter)

    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)


def setup_daily_logger(name, path, level=logging.INFO):
    log_setup = logging.getLogger(name)

    filename = cs.LOG_FOLDER + "/" + str(d.datetime.today().month) + "-" + str(d.datetime.today().day) + '.log'
    fileHandler = logging.FileHandler(os.path.join(path, filename), mode='a')
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler.setFormatter(formatter)

    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)


def clean_filename(filename, replace=' '):
    whitelist = "-_.() %s%s" % (string.ascii_letters, string.digits)
    char_limit = 255

    # replace spaces
    for r in replace:
        filename = filename.replace(r, '_')

    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()

    # keep only whitelisted chars
    cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename) > char_limit:
        print("Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))
    return cleaned_filename[:char_limit]


def export_tabular_to_zip(df, username, title):
    """Exports a dataframe of generated data to an appropriate zip file"""
    full_path = os.path.join(cs.OUTPUT_FOLDER, username)
    safe_mkdir(full_path)
    og_dir = os.getcwd()
    os.chdir(full_path)
    df.to_csv(title + '.txt', index=False)
    with ZipFile(title + '.zip', 'w') as z:
        z.write(title + '.txt')
    os.remove(title + '.txt')
    os.chdir(og_dir)


def create_gen_dict(request_form, directory, username, title, aug=None):
    """Creates a dictionary with keys as dependent variable labels and values as the number of examples pertaining to that label to generate"""
    gen_dict = dict(request_form)
    if aug:
        del gen_dict['download_button']
    for key, value in gen_dict.items():
        gen_dict[key] = 0 if value == '' else int(value)

    assert os.path.exists(os.path.join(directory, username, title))
    if aug:
        filename = cs.GEN_DICT_NAME + ' Additional Data ' + str(aug) + '.pkl'
    else:
        filename = cs.GEN_DICT_NAME + '.pkl'
    with open(os.path.join(directory, username, title, filename), 'wb') as f:
        pkl.dump(gen_dict, f)
