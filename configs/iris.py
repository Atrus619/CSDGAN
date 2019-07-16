# Title of experiment for output directory
EXPERIMENT_NAME = 'Iris_Notebook_Example'

# Split training and test data
TEST_SIZE = 0.5

# Training and CGAN parameters
MANUAL_SEED = 999
NUM_EPOCHS = 2000
CADENCE = 1  # Number of pass-throughs of data set per epoch (generally set to 1, might want to set higher for very tiny data sets)
PRINT_FREQ = 250
EVAL_FREQ = 250
CONT_INPUTS = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']  # Names of features in df that are continuous (not categorical)
INT_INPUTS = []  # Names of features in df that should be integers
DEP_VAR = 'species'  # Name of dependent variable

# Evaluation parameters
EVAL_PARAM_GRID = {'tol': [1e-5],
                   'C': [0.5],
                   'l1_ratio': [0]}
EVAL_FOLDS = 5  # Number of cross-validation folds for evaluation
TEST_RANGES = [75, 150, 300, 600, 1200]  # Various multiples of training set size

TRAINING_PARAMS = {'batch_size': 1000,  # Set to be larger than the data set so that it is always one batch per epoch
                   'shuffle': False,
                   'num_workers': 0}

CGAN_INIT_PARAMS = {'nc': 3,  # Number of output classes
                    'nz': 32,  # Size of noise vector
                    'sched_netG': 1,  # Number of batches to train netG per step (netD gets twice as much data as netG per step by default setting of 1)
                    # Size of hidden layers of subnets
                    'netG_H': 16,
                    'netD_H': 16,
                    # Learning rate for adam optimizer
                    'netG_lr': 2e-4,
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
