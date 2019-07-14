from classes.titanic.CGAN_titanic import CGAN_Generator, CGAN_Discriminator
from utils.utils import *
import random

# Set random seem for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Import data
titanic = pd.read_csv('data/titanic/cleaned.csv')
titanic.head()

# Rearrange so that categorical vars go first
cont_inputs = ['SibSp', 'Parch', 'Fare', 'Age']
cat_inputs = np.array([x for x in titanic.drop(columns='Survived').columns if x not in cont_inputs])
int_inputs = ['SibSp', 'Parch']

cols = titanic.columns.tolist()
cols_start = ['Survived'] + list(cat_inputs)
cols = cols_start + [x for x in cols if x not in cols_start]
titanic = titanic[cols]

# Initialize masks for special data types
cat_mask = ~np.array([x in cont_inputs for x in titanic.drop(columns='Survived').columns])

# Split 50-50 so we can demonstrate the effectiveness of additional data
x_train, x_test, y_train, y_test = train_test_split(titanic.drop(columns='Survived'), titanic['Survived'],
                                                    test_size=445, stratify=titanic['Survived'], random_state=manualSeed)

# Let's convert all of our categorical variables to dummies
le_dict, ohe, x_train, x_test = encode_categoricals_custom(titanic, x_train, x_test, cat_inputs, cat_mask)
preprocessed_cat_mask = create_preprocessed_cat_mask(le_dict=le_dict, x_train=x_train)

# Scale continuous inputs
x_train, scaler = scale_cont_inputs(x_train, preprocessed_cat_mask)
x_train_tensor = torch.tensor(x_train, dtype=torch.float)
y_train_dummies = pd.get_dummies(y_train)
y_train_dummies_tensor = torch.tensor(y_train_dummies.values, dtype=torch.float)

y_test_dummies = pd.get_dummies(y_test)
x_test, _ = scale_cont_inputs(x_test, preprocessed_cat_mask, scaler=scaler)

# Parameters
nz = 64  # Size of generator noise input  # TODO: May need to mess around with this later
H = 32  # Size of hidden network layer
out_dim = x_train.shape[1]  # Size of output
bs = x_train.shape[0]  # Full data set
nc = 2  # 2 different types of label in this problem
num_batches = 1
num_epochs = 10000
print_interval = 1000
exp_name = 'experiments/titanic_3x32_wd_0_uniform_init_only_gen2'
safe_mkdir(exp_name)

# Adam optimizer hyperparameters
lr = 2e-4
beta1 = 0.5
beta2 = 0.999

# Set the device
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# device = 'cpu'

# Instantiate nets
netG = CGAN_Generator(nz=nz, H=H, out_dim=out_dim, nc=nc, bs=bs, device=device, wd=0, cat_mask=preprocessed_cat_mask, le_dict=le_dict).to(device)
netD = CGAN_Discriminator(H=H, out_dim=out_dim, nc=nc, device=device, wd=0).to(device)

# Print classes
print(netG)
print(netD)

# Define labels
real_label = 1
fake_label = 0

# Stratification
train_avg = y_train.mean()
stratify = [1-train_avg, train_avg]

# Train on real data
labels_list = [x for x in y_train_dummies.columns]
param_grid = {'tol': [1e-5],
              'C': [0.5],
              'l1_ratio': [0]}
model_real, score_real = train_test_logistic_reg(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                                 param_grid=param_grid, cv=5, random_state=manualSeed, labels=labels_list)

# For diagnostics
test_range = [bs*2**(x-1) for x in range(5)]
stored_models = []
stored_scores = []
best_score = 0

# Train GAN
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i in range(num_batches):  # Only one batch per epoch since our data is horrifically small
        # Update Discriminator
        # All real batch first
        real_data = x_train_tensor.to(device)  # Format batch (entire data set in this case)
        real_classes = y_train_dummies_tensor.to(device)
        label = torch.full((bs,), real_label, device=device)  # All real labels

        output = netD(real_data, real_classes).view(-1)  # Forward pass with real data through Discriminator
        netD.train_one_step_real(output, label)

        # All fake batch next
        noise = torch.randn(bs, nz, device=device)  # Generate batch of latent vectors
        fake = netG(noise, real_classes)  # Fake batch with netG
        # fake = process_fake_output(fake, le_dict)
        label.fill_(fake_label)
        output = netD(fake.detach(), real_classes).view(-1)
        netD.train_one_step_fake(output, label)
        netD.combine_and_update_opt()
        netD.update_history()

        # Update Generator
        label.fill_(real_label)  # Reverse labels, fakes are real for generator cost
        output = netD(fake, real_classes).view(-1)  # Since D has been updated, perform another forward pass of all-fakes through D
        netG.train_one_step(output, label)
        netG.update_history()

    # Output training stats
    if epoch % print_interval == 0 or (epoch == num_epochs-1):
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch+1, num_epochs, netD.loss.item(), netG.loss.item(), netD.D_x, netD.D_G_z1, netG.D_G_z2))
        with torch.no_grad():
            # Generate various levels of amounts of fake data and test how training compares
            tmp_models, tmp_scores = evaluate_training_progress(test_range=test_range, fake_bs=bs, nz=nz, nc=nc, out_dim=out_dim, netG=netG,
                                                                x_test=x_test, y_test=y_test, manualSeed=manualSeed, labels_list=labels_list,
                                                                param_grid=param_grid, device=device, le_dict=le_dict, stratify=stratify)
        if max(tmp_scores) > best_score:
            best_score = max(tmp_scores)
            torch.save(netG.state_dict(), exp_name + "/best_netG.pt")
        stored_models += tmp_models
        stored_scores += tmp_scores

print("Real data best score:", score_real)
print("GAN best score:", best_score)
if best_score > score_real:
    print("Success! GAN beat real data!")
else:
    print("Failure. GAN did not beat real data...")

# Plot evaluation over time
plot_training_progress(stored_scores=stored_scores, test_range=test_range, num_saves=len(stored_scores) // len(test_range),
                       real_data_score=score_real, save=exp_name)

# Example parsing a model for stats
parse_models(stored_models=stored_models, epoch=0, print_interval=print_interval, test_range=test_range,
             ind=49, x_test=x_test, y_test=y_test, labels=labels_list)

# Output plots
training_plots(netD=netD, netG=netG, num_epochs=num_epochs, save=exp_name)
plot_layer_scatters(netG, title="Generator", save=exp_name)
plot_layer_scatters(netD, title="Discriminator", save=exp_name)

# Load best model
best_netG = CGAN_Generator(nz=nz, H=H, out_dim=out_dim, nc=nc, bs=bs, device=device, wd=0, cat_mask=preprocessed_cat_mask, le_dict=le_dict).to(device)
best_netG.load_state_dict(torch.load(exp_name + "/best_netG.pt"))

# Generate one last set of fake data for diagnostics
genned_data, genned_labels = gen_fake_data(netG=best_netG, bs=test_range[3], nz=nz, nc=nc, labels_list=labels_list, device=device, stratify=stratify)
genned_data = process_fake_output(genned_data, le_dict)
model_fake, score_fake = train_test_logistic_reg(genned_data, genned_labels, x_test, y_test, param_grid=param_grid, cv=5, random_state=manualSeed,
                                                 labels=labels_list)

# Transform fake data back to original form
genned_data_df = fully_process_fake_output(processed_fake_output=genned_data,
                                           genned_labels=genned_labels,
                                           label_name='Survived',
                                           preprocessed_cat_mask=preprocessed_cat_mask,
                                           ohe=ohe,
                                           le_dict=le_dict,
                                           scaler=scaler,
                                           cat_inputs=cat_inputs,
                                           cont_inputs=cont_inputs,
                                           int_inputs=int_inputs)
# Visualize distributions
plot_scatter_matrix(genned_data_df[cont_inputs], "Fake Data", titanic[cont_inputs], scaler=None, save=exp_name)
plot_scatter_matrix(titanic[cont_inputs], "Real Data", titanic[cont_inputs], scaler=None, save=exp_name)

# Visualize categoricals
compare_cats(real=titanic, fake=genned_data_df, x='Sex', y='Survived', hue='Pclass', save=exp_name)

# Conditional scatters
# Class dict
class_dict = {0: ('Died', 'r'),
              1: ('Survived', 'b')}
plot_conditional_scatter(x_real=titanic[cont_inputs].values,
                         y_real=titanic['Survived'].values,
                         x_fake=genned_data_df[cont_inputs].values,
                         y_fake=genned_data_df['Survived'].values,
                         col1=1,
                         col2=2,
                         class_dict=class_dict,
                         og_df=titanic[cont_inputs],
                         scaler=None,
                         alpha=0.25,
                         save=exp_name)

# Conditional densities
plot_conditional_density(x_real=titanic[cont_inputs].values,
                         y_real=titanic['Survived'].values,
                         x_fake=genned_data_df[cont_inputs].values,
                         y_fake=genned_data_df['Survived'].values,
                         col=1,
                         class_dict=class_dict,
                         og_df=titanic[cont_inputs],
                         scaler=None,
                         save=exp_name)


