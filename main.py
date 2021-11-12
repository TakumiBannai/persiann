#!/usr/bin/env python
"""main file.

- run the files;
1. Dataset & dataloader
2. Model the training
3. Evalution

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from dataset import *
from utils import *
from model import *


# Sellect resources（CPU or GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generate sample dataset
n_sample = 100
n_channel, width, height = 1, 32, 32

train_x1, train_x2, train_y = sample_generater(n_sample,
                                               n_channel,
                                               width,
                                               height)

test_x1, test_x2, test_y = sample_generater(n_sample,
                                            n_channel,
                                            width,
                                            height)

# Create dataset class
train_dataset = Mydataset(train_x1, train_x2, train_y)
test_dataset = Mydataset(test_x1, test_x2, test_y)


# Creat data-loader class
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True,
                                           worker_init_fn=worker_init_fn
                                           )

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=32,
                                          shuffle=False,
                                          num_workers=0,
                                          pin_memory=True,
                                          worker_init_fn=worker_init_fn
                                          )


# Model definition
model = Persiann().to(device)

# Loss function and Optimizer method
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
epoch = 10

train_loss, test_loss = [], []
for epoch in tqdm(range(epoch)):
    model, train_l, test_l = train_model(device, model, criterion, optimizer,
                                         train_loader, test_loader)
    train_loss.append(train_l)
    test_loss.append(test_l)
    print("Train loss: {a:.3f}, Test loss: {b:.3f}".format(
          a=train_loss[-1], b=test_loss[-1])
          )


# Evaluation
preds, labels = retrieve_result(device, model, test_loader)
rmse_score = np.sqrt(((preds.reshape(-1) - labels.reshape(-1)) ** 2).mean())
print("RMSE-score: ", rmse_score)
