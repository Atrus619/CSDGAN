# Title of experiment for output directory
EXPERIMENT_NAME = 'mnist_test_200_epochs'

# Desired Train/Validation/Test split
SPLITS = [0.15, 0.05, 0.80]

# Training and CGAN parameters
MANUAL_SEED = 999
NUM_EPOCHS = 200
PRINT_FREQ = 5
EVAL_FREQ = 20
BATCH_SIZE = 128
TRAINING_PARAMS = {'batch_size': BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 6}

CGAN_INIT_PARAMS = {'nc': 10,  # Number of output classes
                    'nz': 64,  # Size of noise vector
                    'num_channels': 1,  # Number of channels in image
                    # Number of feature maps
                    'netG_nf': 128,
                    'netD_nf': 128,
                    # Learning rate for adam optimizer
                    'netG_lr': 2e-4,
                    'netD_lr': 2e-4,
                    'netE_lr': 2e-4,
                    # Betas for adam optimizer
                    'netG_beta1': 0.5,
                    'netG_beta2': 0.999,
                    'netD_beta1': 0.5,
                    'netD_beta2': 0.999,
                    'netE_beta1': 0.5,
                    'netE_beta2': 0.999,
                    # Weight decay for network (regularization)
                    'netG_wd': 0,
                    'netD_wd': 0,
                    'netE_wd': 0,
                    # Fake data generator parameters
                    'fake_data_set_size': 50000,
                    'fake_bs': BATCH_SIZE,
                    # Evaluator parameters
                    'eval_num_epochs': 40,
                    'early_stopping_patience': 3}

