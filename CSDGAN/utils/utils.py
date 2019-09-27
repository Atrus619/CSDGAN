import CSDGAN.utils.constants as cs
import CSDGAN.utils.img_data_loading as cuidl

import os
import pandas as pd
import shutil
import logging
import unicodedata
import string
import datetime as d
from zipfile import ZipFile
import pickle as pkl
from collections import OrderedDict


def get_CGAN(username, title):
    with open(os.path.join(cs.RUN_FOLDER, username, title, 'CGAN.pkl'), 'rb') as f:
        return pkl.load(f)


def get_max_epoch(username, title):
    CGAN = get_CGAN(username, title)
    return CGAN.epoch


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in cs.ALLOWED_EXTENSIONS


def translate_filepath(path):
    return path.replace("|", "/")


def safe_mkdir(path):
    """
    Create a directory if there isn't one already
    Deprecated in favor of os.makedirs(path, exist_ok=True)
    """
    try:
        os.makedirs(path)
    except OSError:
        pass


def new_run_mkdir(directory, username, title):
    """Initialize directories for a new run. Clears out prior uploads with same title."""
    os.makedirs(os.path.join(directory, username), exist_ok=True)
    try:
        shutil.rmtree(os.path.join(directory, username, title))
    except OSError:
        pass
    os.makedirs(os.path.join(directory, username, title), exist_ok=True)
    return os.path.join(directory, username, title)


def parse_tabular_cols(run_id):
    """Parses an uploaded tabular data set and returns a list of columns"""
    run_id = str(run_id)
    filename = os.listdir(os.path.join(cs.UPLOAD_FOLDER, run_id))[0]
    data = pd.read_csv(os.path.join(cs.UPLOAD_FOLDER, run_id, filename), nrows=0)
    return data.columns


def parse_tabular_dep(run_id, dep_var):
    """Parses an uploaded tabular data set and returns a list of unique values for the dependent variable"""
    run_id = str(run_id)
    filename = os.listdir(os.path.join(cs.UPLOAD_FOLDER, run_id))[0]
    data = pd.read_csv(os.path.join(cs.UPLOAD_FOLDER, run_id, filename), usecols=[dep_var])
    return sorted(data[dep_var].unique())


def validate_tabular_choices(dep_var, cont_inputs, int_inputs):
    """
    Checks to see if user choices are consistent with expectations
    Returns failure message if it fails, None otherwise.
    """
    if dep_var in cont_inputs:
        return 'Dependent variable should not be continuous'
    if dep_var in int_inputs:
        return 'Dependent variable should not be integer'
    if any([int_input not in cont_inputs for int_input in int_inputs]):
        return 'Selected integer features should be a subset of selected continuous features'
    return None


def validate_image_choices(dep_var, x_dim, bs, splits, num_epochs, num_channels):
    """
    Checks to see if user choices are consistent with expectations
    Returns failure message if it fails, None otherwise.
    """
    splits = [float(num) for num in splits]

    if sum(splits) != 1:
        return 'Splits must add up to 1 (Currently adding up to ' + str(sum(splits)) + ')'
    if num_channels not in [1, 3]:
        return 'Number of channels in image (' + str(num_channels) + ' found) should be 1 or 3'
    return None


def unzip_and_validate_img_zip(run_id, username, title):
    """
    Validates user submitted data set to ensure that data submitted is a zip file,
    with all images with same label in a folder named with the label name.
    Images should either be the same size, or a specified image size should be provided (all images will be cropped to the same size)
    :param run_id: Run ID associated with this run
    :return: True if validation successful, False otherwise. Also returns a message associated with the failure if failure, and name of file if successful.
    """
    # Check existence of run directory
    run_dir = os.path.join(cs.RUN_FOLDER, username, title)
    if not os.path.exists(run_dir):
        return False, "Run directory does not exist"

    # Perform various checks and unzip data
    path = os.path.join(cs.UPLOAD_FOLDER, str(run_id))
    file = os.listdir(path)[0]
    if not os.path.splitext(file)[1] == '.zip':
        return False, "Image file path passed is not zip"

    zip_ref = ZipFile(os.path.join(path, file), 'r')
    zip_ref.extractall(run_dir)
    zip_ref.close()

    unzipped_path = os.path.join(run_dir, os.path.splitext(file)[0])
    if not os.path.exists(unzipped_path):
        return False, "Image folder not named the same as zip file"
    if not all([os.path.isdir(os.path.join(unzipped_path, x)) for x in os.listdir(unzipped_path)]):
        return False, "Not all files in primary folder are folders"
    return True, os.path.splitext(file)[0]


def parse_image_folder(username, title, file):
    """
    Parses an uploaded image data set and returns various information about its contents
    Returns:
        1. Dimensions of first image found
        2. Number of channels in image
        3. Table with rows of each label per row, and number of instances of that label in the second column
    """
    path = os.path.join(cs.RUN_FOLDER, username, title, file)
    import_gen = cuidl.import_dataset(path=path, bs=cs.IMAGE_DEFAULT_BATCH_SIZE, shuffle=False, incl_paths=True)
    x_dim = cuidl.find_first_img_dim(import_gen=import_gen)
    num_channels = cuidl.find_first_img_num_channels(import_gen=import_gen)
    df, _ = cuidl.scan_image_dataset(path=path)
    summarized_df = df.groupby(by='label').size()

    return x_dim, num_channels, summarized_df


def setup_run_logger(name, username, title, filename='run_log', level=logging.INFO):
    log_setup = logging.getLogger(name)

    os.path.exists(os.path.join(cs.RUN_FOLDER, username, title)), 'Path does not exist: ' + os.path.join(cs.RUN_FOLDER, username, title)

    fileHandler = logging.FileHandler(os.path.join(cs.RUN_FOLDER, username, title, filename + '.log'), mode='a')
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler.setFormatter(formatter)

    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)


def setup_daily_logger(name, path, level=logging.INFO):
    log_setup = logging.getLogger(name)

    assert os.path.exists(cs.LOG_FOLDER), 'Path does not exist: ' + cs.LOG_FOLDER
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
    full_path = os.path.join(cs.OUTPUT_FOLDER, username, title)
    os.makedirs(full_path, exist_ok=True)
    og_dir = os.getcwd()
    os.chdir(full_path)
    df.to_csv(title + '.txt', index=False)
    with ZipFile(title + '.zip', 'w') as z:
        z.write(title + '.txt')
    os.remove(title + '.txt')
    os.chdir(og_dir)


def create_gen_dict(request_form, directory, username, title, aug=None):
    """Creates a dictionary with keys as dependent variable labels and values as the number of examples pertaining to that label to generate"""
    gen_dict = OrderedDict(request_form)
    if aug is not None:
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
