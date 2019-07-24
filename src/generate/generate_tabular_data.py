import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.data_loading import *
import src.constants as cs

"""
Loads a tabular CGAN created by train_tabular_model.py. Generates data based on user specifications.
"""

# Intake arguments
parser = argparse.ArgumentParser('Process a tabular data set')

parser.add_argument('-id', type=str, help='enter identifier for current run', nargs=1, dest='id')

args = parser.parse_args()

run_id = args.id[0]
bs = args.bs[0]

# Check for objects created by train_tabular_model.py
run_dir = os.path.join(cs.RUN_DIR, run_id)
assert os.path.exists(os.path.join(run_dir, 'CGAN.pkl')), \
    "CGAN object not found"
assert os.path.exists(os.path.join(run_dir, 'gen_dict.pkl')), \
    "gen_dict object not found"

# Load in CGAN and gen_dict
with open(os.path.join(run_dir, 'CGAN.pkl'), 'rb') as f:
    CGAN = pkl.load(f)

with open(os.path.join(run_dir, 'gen_dict.pkl'), 'rb') as f:
    gen_dict = pkl.load(f)

# Generate data
df = pd.DataFrame(columns=CGAN.data_gen.dataset.cols)
for i, (dep_class, size) in enumerate(gen_dict.items()):
    stratify = np.eye(CGAN.nc)[i]
    df = pd.concat((df, CGAN.gen_data(size=size, stratify=stratify)), axis=0)

# Output data
df.to_csv(cs.TABULAR_DEFAULT_GENNED_DATA_FILENAME, index=False)
