import os
from collections import OrderedDict
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
               'Benchmarking': 4,
               'Train 0/4': 5,
               'Train 1/4': 6,
               'Train 2/4': 7,
               'Train 3/4': 8,
               'Generating data': 9,
               'Complete': 10,
               'Early Exit': 98,
               'Error': 99,
               'Unavailable': 100}

# Visualizations
VIZ_FOLDER = os.path.join(VOLUME, 'static', 'visualizations')
FILENAME_TRAINING_PLOT = 'training_plot.png'
FILENAME_PLOT_PROGRESS = 'training_progress.png'
FILENAME_netG_LAYER_SCATTERS = 'layer_scatters|Generator_layer_scatters.png'
FILENAME_netD_LAYER_SCATTERS = 'layer_scatters|Discriminator_layer_scatters.png'
FILENAME_HIST_SCATTERS = 'layer_histograms|{net}_epoch_{num}_layer_histograms.png'

# Tabular Only
FILENAME_SCATTER_MATRIX = 'scatter_matrices|{title}_scatter_matrix.png'
FILENAME_COMPARE_CATS = 'compare_cats|{x}_{hue}_cat_comparison.png'
FILENAME_CONDITIONAL_SCATTER = 'conditional_scatters|{col1}_vs_{col2}_conditional_scatter.png'
FILENAME_CONDITIONAL_DENSITY = 'conditional_densities|{col}_conditional_density.png'
MAX_GENNED_DATA_SET_SIZE = 1e6

# Image Only
FILENAME_IMG_GRIDS = 'imgs|Epoch {epoch}.png'
FILENAME_IMG_GIF = 'generation_animation.gif'

# Details about visualizations
AVAILABLE_BASIC_VIZ = [
    {
        'title': FILENAME_TRAINING_PLOT,
        'pretty_title': 'Training Plot',
        'description': 'training_plot_descr'
    },
    {
        'title': FILENAME_PLOT_PROGRESS,
        'pretty_title': 'Training Curve',
        'description': 'Evaluation over training evaluations descr'
    },
    {
        'title': FILENAME_netG_LAYER_SCATTERS,
        'pretty_title': 'Generator Network Layer Scatter Plots',
        'description': 'netG_layer_scatters_descr'
    },
    {
        'title': FILENAME_netD_LAYER_SCATTERS,
        'pretty_title': 'Discriminator Network Layer Scatter Plots',
        'description': 'netD_layer_scatters_descr'
    }
]

AVAILABLE_HIST_VIZ = [
    {
        'title': FILENAME_HIST_SCATTERS,
        'pretty_title': 'Network Layer Weights/Biases Histograms',
        'description': 'Insert_Description_Here'
    }
]

AVAILABLE_TABULAR_VIZ = OrderedDict()
AVAILABLE_TABULAR_VIZ['scatter_matrix'] = {
        'title': FILENAME_SCATTER_MATRIX,
        'pretty_title': 'Scatter Matrix',
        'description': 'INSERT_DESCRIPTION_HERE',
        'url_func': 'viz.gen_scatter_matrix',
        'fake_title': 'Fake Data',
        'real_title': 'Real Data'
}
AVAILABLE_TABULAR_VIZ['compare_cats'] = {
        'title': FILENAME_COMPARE_CATS,
        'pretty_title': 'Categorical Feature Comparison',
        'description': 'INSERT_DESCRIPTION_HERE',
        'url_func': 'viz.gen_compare_cats'
}
AVAILABLE_TABULAR_VIZ['conditional_scatter'] = {
        'title': FILENAME_CONDITIONAL_SCATTER,
        'pretty_title': 'Conditional Scatter Plot',
        'description': 'INSERT_DESCRIPTION_HERE',
        'url_func': 'viz.gen_conditional_scatter'
}
AVAILABLE_TABULAR_VIZ['conditional_density'] = {
        'title': FILENAME_CONDITIONAL_DENSITY,
        'pretty_title': 'Conditional Density Plot',
        'description': 'INSERT_DESCRIPTION_HERE',
        'url_func': 'viz.gen_conditional_density'
}

AVAILABLE_IMAGE_VIZ = OrderedDict()
AVAILABLE_IMAGE_VIZ['img_grid'] = {
    'title': FILENAME_IMG_GRIDS,
    'pretty_title': 'Grid of Generated Images at Specified Epoch',
    'description': 'INSERT_DESCRIPTION_HERE',
    'url_func': 'viz.gen_img_grid'
}
AVAILABLE_IMAGE_VIZ['img_gif'] = {
    'title': FILENAME_IMG_GIF,
    'pretty_title': 'GIF of Generated Images Over Specified Epochs',
    'description': 'INSERT_DESCRIPTION_HERE',
    'url_func': 'viz.gen_img_gif'
}