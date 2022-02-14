import os
import argparse
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
from torch.optim import Adam
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

parser = argparse.ArgumentParser(description='Testing')

parser.add_argument('--model_name', dest='model_name',)
args = parser.parse_args()
assert(args.model_name != None)

results_path = os.path.join(RESULTS_PATH, args.model_name)

with open(os.path.join(results_path, 'vars.pkl'), 'rb') as f:
    data_train = pickle.load(f)['data_train']

img_map_wrapper = ENetWrapper()
if cuda_state:
    img_map_wrapper.model.cuda()

checkpoint = torch.load(os.path.join(results_path, f'img_map_checkpoint_{EPOCHS_IMG_MAP}.pth'))
img_map_wrapper.model.load_state_dict(checkpoint['state_dict'])

################################################################################

flow_wrapper = NICEWrapper()
optim_flow = Adam(flow_wrapper.model.parameters(), lr=LR_FLOW, betas=(BETA1, BETA2))
fsvdd_loss = FSVDDLoss()

if cuda_state:
    flow_wrapper.model.cuda()
    fsvdd_loss.cuda()

fsvdd_loss.init_vars(img_map_wrapper.model, flow_wrapper.model, data_train.dataloader_train_flow)

################################################################################

losses_flow = []

print('Starting flow training...')
flow_wrapper.model.train()

if not os.path.exists(results_path):
    os.makedirs(results_path)

for epoch in range(1, EPOCHS_FLOW + 1):  # First epoch id is 1, not 0

    lr = LR_FLOW * (0.1 ** (epoch // TRAIN_STEP_FLOW))
    for param_group in optim_flow.param_groups:
        param_group['lr'] = lr

    loss_epoch = 0.
    outputs = torch.zeros((NB_IMGS_TRAIN_NI, INPUT_DIM))
    if cuda_state:
        outputs = outputs.cuda()

    pbar = tqdm(enumerate(data_train.dataloader_train_flow))

    for batch_idx, (data, _, idxes) in pbar:

        if cuda_state:
            data = data.cuda()

        optim_flow.zero_grad()
        with torch.no_grad():
            mapping = img_map_wrapper.model.forward(data, False)
        output = flow_wrapper.model(mapping)
        loss = fsvdd_loss(flow_wrapper.model, output)
        loss.backward()
        optim_flow.step()

        fsvdd_loss.update_W(flow_wrapper.model)

        loss_epoch += loss.data.cpu().numpy()
        outputs[idxes] = output.data.clone()

        if batch_idx % LOG_INTERVAL == 0:
            pbar.set_description('Train Epoch: {}/{} (lr: {:.2e}) [{}/{} ({:.0f}%)]. Loss: {:.3f} '.format(epoch,
                                                                                                           EPOCHS_FLOW,
                                                                                                           optim_flow.param_groups[0]['lr'],
                                                                                                           batch_idx * len(data),
                                                                                                           NB_IMGS_TRAIN_NI,
                                                                                                           100. * batch_idx / len(data_train.dataloader_train_flow),
                                                                                                           (loss_epoch / (batch_idx + 1)).item()))

        if batch_idx + 1 >= len(data_train.dataloader_train_flow):
            pbar.set_description('Train Epoch: {}/{} (lr: {:.2e}) [{}/{} ({:.0f}%)]. Loss: {:.3f} '.format(epoch,
                                                                                                           EPOCHS_FLOW,
                                                                                                           optim_flow.param_groups[0]['lr'],
                                                                                                           NB_IMGS_TRAIN_NI,
                                                                                                           NB_IMGS_TRAIN_NI,
                                                                                                           100.,
                                                                                                           (loss_epoch / len(data_train.dataloader_train_flow)).item()))

    fsvdd_loss.update_R(flow_wrapper.model, epoch, outputs)
    del outputs

    losses_flow.append(loss_epoch / len(data_train.dataloader_train_flow))

    torch.save({'epoch': epoch,
                'state_dict': flow_wrapper.model.state_dict()},
               os.path.join(results_path, f'flow_checkpoint_{epoch}.pth'))
    try:
        os.remove(os.path.join(results_path, f'flow_checkpoint_{epoch - CHECKPOINT}.pth'))
    except:
        pass

################################################################################

torch.save({'epoch': epoch,
            'state_dict': flow_wrapper.model.state_dict()},
           os.path.join(results_path, f'flow_checkpoint_{EPOCHS_FLOW}.pth'))

with open(os.path.join(results_path, 'vars.pkl'), 'wb') as f:
    pickle.dump({"data_train": data_train,
                 "fsvdd_loss": fsvdd_loss}, f)

plt.figure(figsize=(14, 12))
plt.plot(range(EPOCHS_FLOW), losses_flow, label='Training loss flow')
plt.legend()

plt.savefig(os.path.join(results_path, 'loss_training_flow.png'))
