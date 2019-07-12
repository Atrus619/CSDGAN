# Title of experiment for output directory
EXPERIMENT_NAME = 'Notebook_Example'

# Desired Train/Test split


# Training and CGAN parameters
MANUAL_SEED = 999
CONT_INPUTS = ['SibSp', 'Parch', 'Fare', 'Age']  # Names of features in df that are continuous (not categorical)
INT_INPUTS = ['SibSp', 'Parch']  # Names of features in df that should be integers
DEP_VAR = 'Survived'  # Name of dependent variable
