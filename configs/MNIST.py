# Title of experiment for output directory
EXPERIMENT_NAME = 'MNIST_Notebook_Example2'

# Desired Train/Validation/Test split
SPLITS = [0.15, 0.05, 0.80]

# Training and CGAN parameters
MANUAL_SEED = 999
NUM_EPOCHS = 400
PRINT_FREQ = 5
EVAL_FREQ = 50
BATCH_SIZE = 128
TRAINING_PARAMS = {'batch_size': BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 6}

CGAN_INIT_PARAMS = {'sched_netG': 2,  # Number of batches to train netG per step (netD gets twice as much data as netG per step by default setting of 1)
                    'label_noise': 0.25,  # Proportion of labels to flip for discriminator (value between 0 and 1)
                    'label_noise_linear_anneal': True,  # Whether to linearly anneal label noise effect
                    'discrim_noise': 0.25,  # Stdev of noise to add to discriminator inputs
                    'discrim_noise_linear_anneal': False,  # Whether to linearly anneal discriminator noise effect
                    'nc': 10,  # Number of output classes
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
                    'early_stopping_patience': 3
                    }
