import os
import argparse
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report

from dataset import TestDataWrapper
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

parser = argparse.ArgumentParser(description='Testing')

parser.add_argument('--model_name', dest='model_name',)
args = parser.parse_args()
assert(args.model_name != None)

################################################################################

results_path = os.path.join(RESULTS_PATH, args.model_name)

with open(os.path.join(results_path, 'vars.pkl'), 'rb') as f:
    dict_vars = pickle.load(f)
    normalize = dict_vars['normalize']
    C = dict_vars['C']
    R = dict_vars['R']
    W = dict_vars['W']

data_test = TestDataWrapper(normalize)
model_wrapper = ENetWrapper()
if cuda_state:
    model_wrapper.model.cuda()

checkpoint = torch.load(os.path.join(results_path, f'checkpoint_{EPOCHS}.pth'))
model_wrapper.model.load_state_dict(checkpoint['state_dict'])

################################################################################

model_wrapper.model.eval()

real_labels = []
pred_labels = []

with torch.no_grad():
    for batch_idx, (data, target, idxes) in tqdm(enumerate(data_test.dataloader_test)):

        if cuda_state:
            data, target = data.cuda(), target.cuda()

        output = model_wrapper.model.forward(data)
        dist = torch.sum((output * (W ** (1 / INPUT_DIM)) - C) ** 2, dim=1)
        scores = dist - R ** 2
        results = [0 if score < 0 else 1 for score in scores]

        # Grab the predictions and the labels into arrays
        real_labels.extend(target.data.cpu().numpy())
        pred_labels.extend(results)

print(classification_report(real_labels, pred_labels))
