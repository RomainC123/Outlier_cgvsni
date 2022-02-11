import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
from torch.optim import SGD
from tqdm import tqdm

from dataset import TrainDataWrapper
from ENet import ENetWrapper
from NICE import NICEWrapper
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

print('Computing transforms...')
data_train = TrainDataWrapper()

################################################################################

img_map_wrapper = ENetWrapper()
optim_img_map = SGD(img_map_wrapper.model.parameters(), lr=LR_IMG_MAP, momentum=MOMENTUM_IMG_MAP, weight_decay=WEIGHT_DECAY_IMG_MAP)
loss_img_map = torch.nn.CrossEntropyLoss()

if cuda_state:
    img_map_wrapper.model.cuda()
    loss_img_map.cuda()

################################################################################

losses_img_map = []

print('Starting image map training...')
img_map_wrapper.model.train()

for epoch in range(EPOCHS_IMG_MAP):  # First epoch id is 1, not 0

    lr = LR_IMG_MAP * (0.1 ** (epoch // TRAIN_STEP_IMG_MAP))
    for param_group in optim_img_map.param_groups:
        param_group['lr'] = lr

    loss_epoch = 0.

    pbar = tqdm(enumerate(data_train.dataloader_train_img_map))

    for batch_idx, (data, target, idxes) in pbar:

        if cuda_state:
            data = data.cuda()
            target = target.cuda()

        optim_img_map.zero_grad()
        output = img_map_wrapper.model.forward(data, True)
        loss = loss_img_map(output, target)
        loss.backward()
        optim_img_map.step()

        loss_epoch += loss.data.cpu().numpy()

        if batch_idx % LOG_INTERVAL == 0:
            pbar.set_description('Train Epoch: {}/{} (lr: {:.2e}) [{}/{} ({:.0f}%)]. Loss: {:.3f} '.format(epoch + 1,
                                                                                                           EPOCHS_IMG_MAP,
                                                                                                           optim_img_map.param_groups[0]['lr'],
                                                                                                           batch_idx * len(data),
                                                                                                           NB_IMGS_TRAIN_NI + NB_IMGS_TRAIN_CG,
                                                                                                           100. * batch_idx / len(data_train.dataloader_train_img_map),
                                                                                                           (loss_epoch / (batch_idx + 1)).item()))

        if batch_idx + 1 >= len(data_train.dataloader_train_img_map):
            pbar.set_description('Train Epoch: {}/{} (lr: {:.2e}) [{}/{} ({:.0f}%)]. Loss: {:.3f} '.format(epoch + 1,
                                                                                                           EPOCHS_IMG_MAP,
                                                                                                           optim_img_map.param_groups[0]['lr'],
                                                                                                           NB_IMGS_TRAIN_NI + NB_IMGS_TRAIN_CG,
                                                                                                           NB_IMGS_TRAIN_NI + NB_IMGS_TRAIN_CG,
                                                                                                           100.,
                                                                                                           (loss_epoch / len(data_train.dataloader_train_img_map)).item()))

    losses_img_map.append(loss_epoch / len(data_train.dataloader_train_img_map))

################################################################################

if not os.path.exists(results_path):
    os.makedirs(results_path)

torch.save({'epoch': epoch,
            'state_dict': img_map_wrapper.model.state_dict()},
           os.path.join(results_path, f'img_map_checkpoint_{EPOCHS_IMG_MAP}.pth'))

with open(os.path.join(results_path, 'vars.pkl'), 'wb') as f:
    pickle.dump({"data_train": data_train}, f)

plt.figure(figsize=(14, 12))
plt.plot(range(EPOCHS_IMG_MAP), losses_img_map, label='Training loss img_map')
plt.legend()

plt.savefig(os.path.join(results_path, 'loss_training_img_map.png'))
