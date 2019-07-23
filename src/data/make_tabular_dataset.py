import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.data_loading import *
import utils.constants as cs
from classes.Tabular.TabularDataset import TabularDataset

"""
Requirements of data set is that it is contained in a flat file and the continuous vs. categorical vs. integer vs. dependent
variables are specified. It should also be specified how to deal with missing data (stretch goal).
"""

# Intake arguments
parser = argparse.ArgumentParser('Process a tabular data set')

parser.add_argument('-id', type=str, help='enter identifier for current run', nargs=1, dest='id')
parser.add_argument('-p', type=str, help='enter path to flat file', nargs=1, dest='path')
parser.add_argument('-dep_var', type=str, help="enter name of dependent variable", nargs=1, dest='dep_var')
parser.add_argument('-cont_inputs', type=str, help="enter list of names of continuous inputs", nargs='+', dest='cont_inputs')
parser.add_argument('-int_inputs', type=str, help="enter list of names of integer inputs", nargs='+', dest='int_inputs')
parser.add_argument('--test_size', type=int, help="enter desired size of test set", nargs=1, dest='test_size')

args = parser.parse_args()

run_id = args.id[0]
path = args.path[0]
dep_var = args.dep_var[0]
cont_inputs = args.cont_inputs
int_inputs = args.int_inputs
test_size = args.test_size[0] if args.test_size is not None else None

# Create directory for downloads
run_dir = os.path.join(cs.RUN_DIR, run_id)
safe_mkdir(run_dir)

# Perform various checks and load in data
assert os.path.splitext(path)[1] in {'.txt', '.csv', '.zip'}, "Path is not zip or flat file"
if os.path.splitext(path)[1] == '.zip':
    zip_ref = zipfile.ZipFile(path, 'r')
    zip_ref.extractall(run_dir)
    zip_ref.close()

    unzipped_path = os.path.join(run_dir, os.path.splitext(os.path.basename(path))[0])
    assert os.path.exists(unzipped_path), \
        "Flat file in zip not named the same as zip file"
    assert os.path.splitext(run_dir)[1] in {'.txt', '.csv'}, \
        "Flat file in zip should be .txt or .csv"
    data = pd.read_csv(run_dir, header=0)
else:
    data = pd.read_csv(path, header=0)

# Initialize data set object
dataset = TabularDataset(df=data,
                         dep_var=dep_var,
                         cont_inputs=cont_inputs,
                         int_inputs=int_inputs,
                         test_size=test_size)

# Pickle relevant objects
model_objects_path = os.path.join(run_dir, cs.MODEL_OBJECTS)
safe_mkdir(model_objects_path)

with open(os.path.join(model_objects_path, "dataset.pkl"), "wb") as f:
    pkl.dump(dataset, f)
