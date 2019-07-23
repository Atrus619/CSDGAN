RUN_DIR = 'model_runs'  # Directory for saving data after processing
MODEL_OBJECTS = 'model_objects'  # Subdirectory of DATA_DIR/run_id for pickling model-related objects
TABULAR_MEM_THRESHOLD = 5e9  # Threshold for determining if entire tabular data set can be stored on GPU (significant speedup)

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

# Tabular output constants
TABULAR_DEFAULT_GENNED_DATA_FILENAME = 'genned_data.txt'
