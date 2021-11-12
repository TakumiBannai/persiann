#!/usr/bin/env python
"""Utility.

- fix random seed
- data preprocessing

"""

import torch
import random
import numpy as np



def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def sample_generater(n_sample, n_channel, width, height):
    x1 = torch.rand([n_sample, n_channel, width, height])
    x2 = torch.rand([n_sample, n_channel, width, height])
    y = torch.rand([n_sample, n_channel, width, height])
    return x1, x2, y

def train_model(device, model, criterion, optimizer, train_loader, test_loader):
    # Train loop ----------------------------
    model.train()
    train_batch_loss = []
    for x1, x2, y in train_loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x1, x2)
        loss = criterion(y, y_hat)
        loss.backward()
        optimizer.step()
        train_batch_loss.append(loss.item())

    # Test(val) loop ----------------------------
    model.eval()
    test_batch_loss = []
    with torch.no_grad():  # 勾配を計算なし
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y_hat = model(x1, x2)
            loss = criterion(y, y_hat)
            test_batch_loss.append(loss.item())

    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)


def retrieve_result(device, model, dataloader):
    model.eval()
    preds = []
    labels = []
    # Retreive prediction and labels
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y_hat = model(x1, x2)
            # Collect data
            preds.append(y_hat)
            labels.append(y)
    # Flatten
    preds = torch.cat(preds, axis=0)
    labels = torch.cat(labels, axis=0)
    # Returns as numpy (if GPU)
    # preds = preds.cpu().detach().numpy()
    # labels = labels.cpu().detach().numpy()
    return preds, labels
