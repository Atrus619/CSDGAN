from scripts.Utils.utils import *

# Output plots
training_plots(netD=netD, netG=netG, num_epochs=num_epochs)

# Train various models with real/fake data.
y_test_dummies = pd.get_dummies(y_test)
print("Dummy columns match?", all(y_train_dummies.columns == y_test_dummies.columns))
x_test = scaler.transform(x_test)
labels_list = [x for x in y_train_dummies.columns]
param_grid = {'tol': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
              'C': [0.5, 0.75, 1, 1.25],
              'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}

model_real, score_real = train_test_logistic_reg(x_train, y_train, x_test, y_test, param_grid=param_grid, cv=5, random_state=manualSeed, labels=labels_list)

# Generate various levels of amounts of fake data and test how training compares
test_range = [75, 150, 300, 600, 1200]
fake_bs = bs
fake_models = []
fake_scores = []
for size in test_range:
    num_batches = size // fake_bs + 1
    genned_data = np.empty((0, out_dim))
    genned_labels = np.empty(0)
    rem = size
    while rem > 0:
        curr_size = min(fake_bs, rem)
        noise = torch.randn(curr_size, nz, device=device)
        fake_labels, output_labels = gen_labels(size=curr_size, num_classes=nc, labels_list=labels_list)
        fake_labels = fake_labels.to(device)
        rem -= curr_size
        fake_data = netG(noise, fake_labels).cpu().detach().numpy()
        genned_data = np.concatenate((genned_data, fake_data))
        genned_labels = np.concatenate((genned_labels, output_labels))
    print("For size of:", size)
    model_fake_tmp, score_fake_tmp = train_test_logistic_reg(genned_data, genned_labels, x_test, y_test,
                                                             param_grid=param_grid, cv=5, random_state=manualSeed, labels=labels_list)
    fake_models.append(model_fake_tmp)
    fake_scores.append(score_fake_tmp)

# Visualize distributions
iris_plot_scatters(genned_data, genned_labels, "Fake Data", scaler, alpha=0.5)  # Fake data
iris_plot_scatters(iris.drop(columns='species'), np.array(iris.species, alpha=0.5), "Full Real Data Set")  # All real data
iris_plot_scatters(x_train, np.array(y_train), "Training Data", scaler, alpha=0.5)  # Real train data
iris_plot_scatters(x_test, np.array(y_test), "Testing Data", scaler, alpha=0.5)  # Real test data

iris_plot_densities(genned_data, genned_labels, "Fake Data", scaler)  # Fake data
iris_plot_densities(iris.drop(columns='species'), np.array(iris.species), "Full Real Data Set")  # All real data
iris_plot_densities(x_train, np.array(y_train), "Training Data", scaler)  # Real train data
iris_plot_densities(x_test, np.array(y_test), "Testing Data", scaler)  # Real test data

# Visualize output of tests
fake_data_training_plots(real_range=75, score_real=score_real, test_range=test_range, fake_scores=fake_scores)