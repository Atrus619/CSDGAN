import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.data_loading import *
import utils.constants as cs
from classes.Tabular.TabularCGAN import TabularCGAN

"""
Trains a Tabular CGAN on the data preprocessed by make_tabular_dataset.py. Loads best generator and pickles CGAN for predictions
"""

# Intake arguments
parser = argparse.ArgumentParser('Process a tabular data set')

parser.add_argument('-id', type=str, help='enter identifier for current run', nargs=1, dest='id')
parser.add_argument('-bs', type=int, help='enter batch size for loading', nargs=1, dest='bs')

args = parser.parse_args()

run_id = args.id[0]
bs = args.bs[0]

# Check for objects created by make_tabular_dataset.py
run_dir = os.path.join(cs.RUN_DIR, run_id)
assert os.path.exists(os.path.join(run_dir, 'dataset.pkl')), \
    "Data set object not found"

# Load data set and create CGAN object
with open(os.path.join(run_dir, cs.MODEL_OBJECTS, "dataset.pkl"), 'rb') as f:
    dataset = pkl.load(f)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

if len(pkl.dumps(dataset, -1)) < cs.TABULAR_MEM_THRESHOLD:
    dataset.to_dev(device)

data_gen = data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)

CGAN = TabularCGAN(data_gen=data_gen,
                   device=device,
                   path=run_dir,
                   seed=None,
                   eval_param_grid=cs.TABULAR_EVAL_PARAM_GRID,
                   eval_folds=cs.TABULAR_EVAL_FOLDS,
                   test_ranges=[dataset.x_train.shape[0]*2**x for x in range(5)],
                   eval_stratify=dataset.eval_stratify,
                   nc=len(dataset.labels_list),
                   nz=cs.TABULAR_DEFAULT_NZ,
                   sched_netG=cs.TABULAR_DEFAULT_SCHED_NETG,
                   netG_H=cs.TABULAR_DEFAULT_NETG_H,
                   netD_H=cs.TABULAR_DEFAULT_NETD_H,
                   **cs.TABULAR_CGAN_INIT_PARAMS)

# Train
CGAN.train_gan(num_epochs=cs.TABULAR_DEFAULT_NUM_EPOCHS,
               cadence=cs.TABULAR_DEFAULT_CADENCE,
               print_freq=cs.TABULAR_DEFAULT_PRINT_FREQ,
               eval_freq=cs.TABULAR_DEFAULT_EVAL_FREQ)

# Load best-performing GAN and pickle CGAN to main directory
CGAN.load_netG(best=True)

with open(os.path.join(run_dir, 'CGAN.pkl'), 'wb') as f:
    pkl.dump(CGAN, f)
