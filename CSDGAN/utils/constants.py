import os
from CSDGAN.fake_create_app import fake_create_app

TABULAR_MEM_THRESHOLD = 1024 ** 3 * 5  # Threshold for determining if entire tabular data set can be stored on GPU (significant speedup)

# Evaluation parameters for tabular data sets
TABULAR_EVAL_PARAM_GRID = {'tol': [1e-5],
                           'C': [0.5],
                           'l1_ratio': [0]}
TABULAR_EVAL_FOLDS = 5  # Number of cross-validation folds for evaluation

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
                            }

# Specific tabular initialization parameters
TABULAR_DEFAULT_NZ = 64
TABULAR_DEFAULT_SCHED_NETG = 1
TABULAR_DEFAULT_NETG_H = 32
TABULAR_DEFAULT_NETD_H = 32

# Tabular training parameters
TABULAR_DEFAULT_NUM_EPOCHS = 10000
TABULAR_DEFAULT_CADENCE = 1
TABULAR_DEFAULT_PRINT_FREQ = 250
TABULAR_DEFAULT_EVAL_FREQ = 250
TABULAR_MAX_NUM_EPOCHS = 100000
TABULAR_DEFAULT_TEST_SIZE = 0.2
TABULAR_DEFAULT_BATCH_SIZE = 1000

# App constants
TESTING = False
DEBUG = True
app = fake_create_app()
# TODO: This may cause issues...
DATABASE = os.path.join(app.instance_path, 'csdgan.sqlite')
UPLOAD_FOLDER = os.path.join(app.root_path, 'incoming_raw_data')  #'/home/aj/PycharmProjects/Synthetic_Data_GAN_Capstone/downloads/incoming_raw_data'
RUN_FOLDER = os.path.join(app.root_path, 'runs')  #'/home/aj/PycharmProjects/Synthetic_Data_GAN_Capstone/runs'
OUTPUT_FOLDER = os.path.join(app.root_path, 'genned_data')  #'/home/aj/PycharmProjects/Synthetic_Data_GAN_Capstone/CSDGAN/genned_data'
LOG_FOLDER = os.path.join(app.root_path, 'logs')  #'/home/aj/PycharmProjects/Synthetic_Data_GAN_Capstone/logs'

ALLOWED_EXTENSIONS = {'txt', 'csv', 'zip'}
MAX_CONTENT_LENGTH = 1024 ** 3 * 16  # Maximum data size of 16GB
AVAILABLE_FORMATS = ['Tabular', 'Image']
REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'

# Run constants
GEN_DICT_NAME = 'gen_dict'
MAX_EXAMPLE_PER_CLASS = 10000

# Run statuses
STATUS_DICT = {'Not started': 1,
               'Preprocessing data': 2,
               'Train 0/4': 3,
               'Train 1/4': 4,
               'Train 2/4': 5,
               'Train 3/4': 6,
               'Generating data': 7,
               'Complete': 8,
               'Error': 99,
               'Unavailable': 100}
