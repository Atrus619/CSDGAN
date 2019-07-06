import torch.nn as nn
import torch
from models.NetUtils import NetUtils
import torch.optim as optim
import numpy as np


# Evaluator class
class CGAN_Evaluator(nn.Module, NetUtils):
    def __init__(self, train_gen, val_gen, test_gen, device, num_channels, nc, lr, beta1, beta2, wd):
        super().__init__()

        # Generators
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen

        self.device = device

        # Layers
        self.cn1 = nn.Conv2d(in_channels=num_channels, out_channels=10, kernel_size=5, stride=1, padding=2, bias=True)
        self.cn1_bn = nn.BatchNorm2d(10)
        self.cn2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2, bias=True)
        self.cn2_bn = nn.BatchNorm2d(20)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=14 * 14 * 20, out_features=64)
        self.output = nn.Linear(in_features=64, out_features=nc)

        # Activations
        self.do2d = nn.Dropout2d(0.2)
        self.do1d = nn.Dropout(0.2)
        self.act = nn.LeakyReLU(0.2)

        # Loss and Optimizer
        self.loss = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # Initialize weights
        self.weights_init()

        # Record history
        self.train_losses = []
        self.val_losses = []
        self.val_acc = []

    def forward(self, x):
        x = self.do2d(self.act(self.cn1_bn(self.cn1(x))))
        x = self.do2d(self.act(self.cn2_bn(self.cn2(x))))
        x = self.mp(x)
        x = x.view(-1, 14 * 14 * 20)
        x = self.do1d(self.act(self.fc1(x)))
        return self.output(x)  # No softmax activation needed because it is built into CrossEntropyLoss in pytorch

    def train_one_epoch(self):
        self.train()
        train_loss = 0
        running_count = 0
        for batch, labels in self.train_gen:
            # Forward pass
            batch, labels = batch.to(self.device), labels.to(self.device)
            self.zero_grad()
            fwd = self.forward(batch)
            # Calculate loss and update optimizer
            label_ind = torch.argmax(labels, -1)
            self.loss = self.loss_fn(fwd, label_ind)
            self.loss.backward()
            self.opt.step()
            # Update running totals
            train_loss += self.loss
            running_count += len(batch)
        return train_loss / running_count

    def eval_once(self, gen):
        self.eval()
        val_loss = 0
        num_correct = torch.zeros(1)
        running_count = 0
        with torch.no_grad():
            for batch, labels in gen:
                # Forward pass
                batch, labels = batch.to(self.device), labels.to(self.device)
                fwd = self.forward(batch)
                # Calculate loss and accuracy, and update running totals
                label_ind = torch.argmax(labels, -1)
                val_loss += self.loss_fn(fwd, label_ind)
                num_correct += sum(torch.argmax(fwd, -1) == label_ind)
                running_count += len(batch)
        return val_loss / running_count, num_correct / running_count

    def train_evaluator(self, num_epochs, eval_freq, es=None):
        for epoch in range(num_epochs):
            total_loss = self.train_one_epoch()
            self.train_losses.append(total_loss.item())

            if epoch % eval_freq == 0 or (epoch == num_epochs - 1):
                total_loss, total_acc = self.eval_once(self.val_gen)
                self.val_losses.append(total_loss.item())
                self.val_acc.append(total_acc.item())

                if es:
                    if np.argmin(self.val_losses) < (epoch - es + 1):
                        return True  # Exit early
        return True
