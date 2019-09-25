import os
import pandas as pd
from CSDGAN.fake_create_app import fake_create_app

basedir = os.path.abspath(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(basedir, '.env'))

except ModuleNotFoundError:
    pass

DOCKERIZED = int(os.environ.get('DOCKERIZED')) if os.environ.get('DOCKERIZED') is not None else 0

TABULAR_MEM_THRESHOLD = 1024 ** 3 * 5  # Threshold for determining if entire tabular data set can be stored on GPU (significant speedup)

# Evaluation parameters for tabular data sets
TABULAR_EVAL_PARAM_GRID = {'tol': [1e-5],
                           'C': [0.5],
                           'l1_ratio': [0]}
TABULAR_EVAL_FOLDS = 5  # Number of cross-validation folds for evaluation

# Specific tabular initialization parameters
TABULAR_CGAN_INIT_PARAMS = {'netG_lr': 2e-4,  # Learning rate for adam optimizer
                            'netD_lr': 2e-4,
                            # Betas for adam optimizer
                            'netG_beta1': 0.5,
                            'netD_beta1': 0.5,
                            'netG_beta2': 0.999,
                            'netD_beta2': 0.999,
                            # Weight decay for network (regularization)
                            'netG_wd': 0,
                            'netD_wd': 0,
                            'label_noise': 0.0,  # Proportion of labels to flip for discriminator (value between 0 and 1)
                            'label_noise_linear_anneal': False,  # Whether to linearly anneal label noise effect
                            'discrim_noise': 0.0,  # Stdev of noise to add to discriminator inputs
                            'discrim_noise_linear_anneal': False,  # Whether to linearly anneal discriminator noise effect
                            # Various initialization parameters
                            'nz': 64,
                            'sched_netG': 1,
                            'netG_H': 32,
                            'netD_H': 32
                            }

# Tabular training parameters
TABULAR_DEFAULT_NUM_EPOCHS = 10000
TABULAR_DEFAULT_CADENCE = 1
TABULAR_DEFAULT_PRINT_FREQ = 250
TABULAR_DEFAULT_EVAL_FREQ = 250
TABULAR_MAX_NUM_EPOCHS = 100000
TABULAR_DEFAULT_TEST_SIZE = 0.2
TABULAR_DEFAULT_BATCH_SIZE = 1000

# Specific image initialization parameters
IMAGE_CGAN_INIT_PARAMS = {'netG_lr': 2e-4,  # Learning rate for adam optimizer
                          'netD_lr': 2e-4,
                          'netE_lr': 2e-4,
                          # Betas for adam optimizer
                          'netG_beta1': 0.5,
                          'netD_beta1': 0.5,
                          'netE_beta1': 0.5,
                          'netG_beta2': 0.999,
                          'netD_beta2': 0.999,
                          'netE_beta2': 0.999,
                          # Weight decay for network (regularization)
                          'netG_wd': 0,
                          'netD_wd': 0,
                          'netE_wd': 0,
                          'label_noise': 0.25,  # Proportion of labels to flip for discriminator (value between 0 and 1)
                          'label_noise_linear_anneal': True,  # Whether to linearly anneal label noise effect
                          'discrim_noise': 0.25,  # Stdev of noise to add to discriminator inputs
                          'discrim_noise_linear_anneal': True,  # Whether to linearly anneal discriminator noise effect
                          # Various initialization parameters
                          'nz': 64,
                          'sched_netG': 2,
                          'netG_nf': 128,
                          'netD_nf': 128,
                          # Fake data generator parameters
                          'fake_data_set_size': 50000,
                          # Evaluator parameters
                          'eval_num_epochs': 40,
                          'early_stopping_patience': 3
                          }

# Image training parameters
IMAGE_DEFAULT_NUM_EPOCHS = 400
IMAGE_DEFAULT_TRAIN_VAL_TEST_SPLITS = [0.80, 0.10, 0.10]
IMAGE_DEFAULT_BATCH_SIZE = 128
IMAGE_DEFAULT_PRINT_FREQ = 5
IMAGE_DEFAULT_EVAL_FREQ = 50
IMAGE_DEFAULT_CLASS_NAME = 'Image Class'

# Max Image parameters
IMAGE_MAX_X_DIM = 1080
IMAGE_MAX_BS = 512
IMAGE_MAX_NUM_EPOCHS = 1000

# App constants
app = fake_create_app()

TESTING = False
DEBUG = False
VOLUME = '/MyDataVolume' if DOCKERIZED else app.root_path
UPLOAD_FOLDER = os.path.join(VOLUME, 'incoming_raw_data')
RUN_FOLDER = os.path.join(VOLUME, 'runs')
OUTPUT_FOLDER = os.path.join(VOLUME, 'genned_data')
LOG_FOLDER = os.path.join(VOLUME, 'logs')

ALLOWED_EXTENSIONS = {'txt', 'csv', 'zip'}
MAX_CONTENT_LENGTH = 1024 ** 3 * 16  # Maximum data size of 16GB
AVAILABLE_FORMATS = ['Tabular', 'Image']

# Run constants
GEN_DICT_NAME = 'gen_dict'
MAX_EXAMPLE_PER_CLASS = 10000

# Run statuses - Make sure to check schema.sql as well if changes are made
STATUS_DICT = {'Not started': 1,
               'Kicked off': 2,
               'Preprocessing data': 3,
               'Train 0/4': 4,
               'Train 1/4': 5,
               'Train 2/4': 6,
               'Train 3/4': 7,
               'Generating data': 8,
               'Complete': 9,
               'Early Exit': 98,
               'Error': 99,
               'Unavailable': 100}

# Filenames
VIZ_FOLDER = os.path.join(VOLUME, 'static', 'visualizations')
FILENAME_TRAINING_PLOT = 'training_plot.png'

# Visualizations
AVAILABLE_TABULAR_VIZ = [
    {
        'title': FILENAME_TRAINING_PLOT,
        'pretty_title': 'Training Plot',
        'description': 'training_plot_descr'
    }# ,
    # {
    #     'title': 'layer_scatters',
    #     'pretty_title': 'Layer Scatters',
    #     'description': 'layer_scatters_descr'
    # }
]
AVAILABLE_IMAGE_VIZ = [
    {
        'title': FILENAME_TRAINING_PLOT,
        'pretty_title': 'Training Plot',
        'description': 'training_plot_descr'
    }# ,
    # {
    #     'title': 'layer_scatters',
    #     'pretty_title': 'Layer Scatters',
    #     'description': 'layer_scatters_descr'
    # }
]