import configs.titanic as cfg
import random
import torch
from utils.utils import safe_mkdir, define_cat_inputs, reorder_cols
from utils.data_loading import load_processed_dataset
import os

# Set random seem for reproducibility
print("Random Seed: ", cfg.MANUAL_SEED)
random.seed(cfg.MANUAL_SEED)
torch.manual_seed(cfg.MANUAL_SEED)

# Ensure directory exists for outputs
exp_path = os.path.join("experiments", cfg.EXPERIMENT_NAME)
safe_mkdir(exp_path)

# Import data
titanic = load_processed_dataset('titanic')

# Automatically determine these parameters and complete preprocessing
cat_inputs = define_cat_inputs(df=titanic, dep_var=cfg.DEP_VAR, cont_inputs=cfg.CONT_INPUTS)
titanic = reorder_cols(df=titanic, dep_var=cfg.DEP_VAR, cat_inputs=cat_inputs)
