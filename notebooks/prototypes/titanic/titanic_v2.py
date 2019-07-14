import configs.titanic as cfg
from utils.utils import *
from utils.data_loading import load_processed_dataset
import os
from classes.titanic.CGAN import CGAN
from classes.TabularDataset import TabularDataset
from torch.utils import data

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
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
out_dim = titanic.shape[1]
labels_list = list(titanic[cfg.DEP_VAR].unique())
eval_stratify = [1 - titanic[cfg.DEP_VAR].mean(), titanic[cfg.DEP_VAR].mean()]

# Instantiate data set and generator
dataset = TabularDataset(df=titanic,
                         dep_var=cfg.DEP_VAR,
                         cont_inputs=cfg.CONT_INPUTS,
                         int_inputs=cfg.INT_INPUTS,
                         test_size=cfg.TEST_SIZE,
                         seed=cfg.MANUAL_SEED)
data_gen = data.DataLoader(dataset, **cfg.TRAINING_PARAMS)

# Define GAN
CGAN = CGAN(data_gen=data_gen,
            device=device,
            path=exp_path,
            seed=cfg.MANUAL_SEED,
            eval_param_grid=cfg.EVAL_PARAM_GRID,
            eval_folds=cfg.EVAL_FOLDS,
            test_ranges=cfg.TEST_RANGES,
            eval_stratify=eval_stratify,
            **cfg.CGAN_INIT_PARAMS)

# Eval on real data
score_real = train_test_logistic_reg(x_train=dataset.x_train_arr,
                                     y_train=dataset.y_train_arr,
                                     x_test=dataset.x_test_arr,
                                     y_test=dataset.y_test_arr,
                                     param_grid=cfg.EVAL_PARAM_GRID,
                                     cv=cfg.EVAL_FOLDS,
                                     random_state=cfg.MANUAL_SEED,
                                     labels_list=labels_list,
                                     verbose=True)

# Train GAN
CGAN.train_gan(num_epochs=cfg.NUM_EPOCHS, cadence=cfg.CADENCE, print_freq=cfg.PRINT_FREQ, eval_freq=cfg.EVAL_FREQ)

# Visualizations
CGAN.plot_progress(benchmark_acc=score_real, show=True, save=exp_path)
CGAN.plot_training_plots(show=True, save=exp_path)
CGAN.netG.plot_layer_scatters(title="Generator", show=True, save=exp_path)
CGAN.netD.plot_layer_scatters(title="Discriminator", show=True, save=exp_path)
CGAN.netG.plot_layer_hists(title="Generator", show=True, save=exp_path)
CGAN.netD.plot_layer_hists(title="Discriminator", show=True, save=exp_path)

genned_df = CGAN.gen_data(size=1000, stratify=eval_stratify)
plot_scatter_matrix(X=genned_df[cfg.CONT_INPUTS], title="Fake Data", og_df=titanic[cfg.CONT_INPUTS], scaler=None, save=exp_path)
plot_scatter_matrix(X=genned_df[cfg.CONT_INPUTS], title="Fake Data", og_df=titanic[cfg.CONT_INPUTS], scaler=None, save=exp_path)

compare_cats(real=titanic, fake=genned_df, x='Sex', y='Survived', hue='Pclass', save=exp_path)

class_dict = {0: ('Died', 'r'),
              1: ('Survived', 'b')}

plot_conditional_scatter(x_real=titanic[cfg.CONT_INPUTS].values,
                         y_real=titanic[cfg.DEP_VAR].values,
                         x_fake=genned_df[cfg.CONT_INPUTS].values,
                         y_fake=genned_df[cfg.DEP_VAR].values,
                         col1=1,
                         col2=2,
                         class_dict=class_dict,
                         og_df=titanic[cfg.CONT_INPUTS],
                         scaler=None,
                         alpha=0.25,
                         save=exp_path)

plot_conditional_density(x_real=titanic[cfg.CONT_INPUTS].values,
                         y_real=titanic[cfg.DEP_VAR].values,
                         x_fake=genned_df[cfg.CONT_INPUTS].values,
                         y_fake=genned_df[cfg.DEP_VAR].values,
                         col=1,
                         class_dict=class_dict,
                         og_df=titanic[cfg.CONT_INPUTS],
                         scaler=None,
                         save=exp_path)
