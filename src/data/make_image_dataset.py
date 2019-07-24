import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import src.constants as cs
from utils.data_loading import *

"""
Requirements of image data set is that it should be a single zip with all images with same label in a folder named with the label name
Images should either be the same size, or a specified image size should be provided (all images will be cropped to the same size)
This file accomplishes the following:
    1. Accepts path to a zip file, a desired image size (optional, else first image dim will be used), batch size, and train/val/test splits
    2. Checks to ensure file is a zip, and that unzipped file is structured properly
    3. Unzips files
    4. Splits data into train/val/test splits via stratified sampling and moves into corresponding folders
    5. Deletes original unzipped images
    6. Pickles label encoder, one hot encoder, resulting image size, and all three generators
"""

# Intake arguments
parser = argparse.ArgumentParser('Process an image data set')

parser.add_argument('-id', type=str, help='enter identifier for current run', nargs=1, dest='id')
parser.add_argument('-p', type=str, help='enter path to image zip', nargs=1, dest='path')
parser.add_argument('-bs', type=int, help='enter batch size for loading', nargs=1, dest='bs')
parser.add_argument('--x_dim', type=tuple, help="enter desired image size, otherwise first image's size will be used", nargs=1, dest='x_dim')
parser.add_argument('--splits', type=tuple, help="enter desired train/val/test splits", nargs=1, dest='splits')

args = parser.parse_args()

run_id = args.id[0]
path = args.path[0]
bs = args.bs[0]
x_dim = args.x_dim[0] if args.x_dim is not None else None
splits = args.splits if args.splits is not None else None

# img_zip = os.path.join("downloads", "fruits.zip")
# bs = 128
# x_dim = None

assert os.path.splitext(path)[1] == '.zip', "Image file path passed is not zip"

# Create directory for downloads
run_dir = os.path.join(cs.RUN_DIR, run_id)
safe_mkdir(run_dir)

# Unzip folders
zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall(run_dir)
zip_ref.close()

# Preprocess data
unprocessed_img_path = os.path.join(run_dir, os.path.splitext(os.path.basename(path))[0])
assert os.path.exists(unprocessed_img_path), \
    "Image folder not named the same as zip file"
assert all([os.path.isdir(os.path.join(unprocessed_img_path, x)) for x in os.listdir(unprocessed_img_path)]), \
    "Not all files in main folder are folders"

import_gen = import_dataset(path=unprocessed_img_path, bs=bs, shuffle=False)

le, ohe, x_dim = preprocess_imported_dataset(path=unprocessed_img_path, import_gen=import_gen,
                                             splits=splits, x_dim=x_dim)

# Create data loader for each component of data set
train_gen = import_dataset(os.path.join(unprocessed_img_path, 'train'), bs=bs, shuffle=True)
val_gen = import_dataset(os.path.join(unprocessed_img_path, 'val'), bs=bs, shuffle=False)
test_gen = import_dataset(os.path.join(unprocessed_img_path, 'test'), bs=bs, shuffle=False)

# Pickle relevant objects
model_objects_path = os.path.join(run_dir, cs.MODEL_OBJECTS)
safe_mkdir(model_objects_path)

with open(os.path.join(model_objects_path, "le.pkl"), "wb") as f:
    pkl.dump(le, f)

with open(os.path.join(model_objects_path, "ohe.pkl"), "wb") as f:
    pkl.dump(le, ohe)

with open(os.path.join(model_objects_path, "x_dim.pkl"), "wb") as f:
    pkl.dump(le, x_dim)

with open(os.path.join(model_objects_path, "train_gen.pkl"), "wb") as f:
    pkl.dump(le, train_gen)

with open(os.path.join(model_objects_path, "val_gen.pkl"), "wb") as f:
    pkl.dump(le, val_gen)

with open(os.path.join(model_objects_path, "test_gen.pkl"), "wb") as f:
    pkl.dump(le, test_gen)
