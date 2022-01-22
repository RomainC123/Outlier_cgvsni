import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
from torch.optim import Adam
from tqdm import tqdm

from dataset import TrainDataWrapper
from ENet import ENetWrapper
from loss import FSVDDLoss
from params.constants import *
from params.paths import RESULTS_PATH

import warnings
warnings.filterwarnings('ignore')

cuda_state = torch.cuda.is_available()

seed = SEED
np.random.seed(seed)  # Fixes the dataset, but not the training behavior

################################################################################

results_path = os.path.join(RESULTS_PATH, '{date:%d-%m-%Y_%H:%M:%S}'.format(date=datetime.now()))

data_train = TrainDataWrapper()
model_wrapper = ENetWrapper()
optim = Adam(model_wrapper.model.parameters(), lr=LR, betas=(BETA1, BETA2))
fsvdd_loss = FSVDDLoss()
if cuda_state:
    model_wrapper.model.cuda()
    fsvdd_loss.cuda()

################################################################################

print('Starting training...')
model_wrapper.model.train()
fsvdd_loss.init_vars(model_wrapper.model, data_train.dataloader_train)

losses = []

for epoch in range(1, EPOCHS + 1):  # First epoch id is 1, not 0

    loss_epoch = 0.
    outputs = torch.zeros((NB_IMGS_TRAIN, INPUT_DIM))
    if cuda_state:
        outputs = outputs.cuda()

    pbar = tqdm(enumerate(data_train.dataloader_train))

    for batch_idx, (data, _, idxes) in pbar:

        if cuda_state:
            data = data.cuda()

        optim.zero_grad()
        output = model_wrapper.model.forward(data)
        loss = fsvdd_loss(model_wrapper.model, output)
        loss.backward()
        optim.step()

        fsvdd_loss.update_W(model_wrapper.model)

        loss_epoch += loss.data.cpu().numpy()
        outputs[idxes] = output.data.clone()

        if batch_idx % LOG_INTERVAL == 0:
            pbar.set_description('Train Epoch: {}/{} (lr: {:.2e}) [{}/{} ({:.0f}%)]. Loss: {:.3f} '.format(epoch,
                                                                                                           EPOCHS,
                                                                                                           optim.param_groups[0]['lr'],
                                                                                                           batch_idx * len(data),
                                                                                                           NB_IMGS_TRAIN,
                                                                                                           100. * batch_idx / len(data_train.dataloader_train),
                                                                                                           (loss_epoch / (batch_idx + 1)).item()))

        if batch_idx + 1 >= len(data_train.dataloader_train):
            pbar.set_description('Train Epoch: {}/{} (lr: {:.2e}) [{}/{} ({:.0f}%)]. Loss: {:.3f} '.format(epoch,
                                                                                                           EPOCHS,
                                                                                                           optim.param_groups[0]['lr'],
                                                                                                           NB_IMGS_TRAIN,
                                                                                                           NB_IMGS_TRAIN,
                                                                                                           100.,
                                                                                                           (loss_epoch / len(data_train.dataloader_train)).item()))

    fsvdd_loss.update_R(model_wrapper.model, epoch, outputs)
    del outputs

    losses.append(loss_epoch / len(data_train.dataloader_train))

################################################################################

if not os.path.exists(results_path):
    os.makedirs(results_path)

torch.save({'epoch': epoch,
            'state_dict': model_wrapper.model.state_dict()},
           os.path.join(results_path, f'checkpoint_{epoch}.pth'))

with open(os.path.join(results_path, 'vars.pkl'), 'wb') as f:
    pickle.dump({"normalize": data_train.normalize,
                 "C": fsvdd_loss.C,
                 "R": fsvdd_loss.R,
                 "W": fsvdd_loss.W}, f)

plt.figure(figsize=(14, 12))
plt.plot(range(EPOCHS), losses, label='Training loss')
plt.legend()

plt.savefig(os.path.join(results_path, 'loss_training.png'))
