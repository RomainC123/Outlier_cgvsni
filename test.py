import os
import argparse
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from dataset import TestDataWrapper
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

################################################################################

results_path = os.path.join(RESULTS_PATH, args.model_name)

with open(os.path.join(results_path, 'vars.pkl'), 'rb') as f:
    dict_vars = pickle.load(f)
    train_ni_idx = dict_vars['train_ni_idx']
    train_cg_idx = dict_vars['train_cg_idx']
    normalize_img_map = dict_vars['normalize_img_map']
    normalize_flow = dict_vars['normalize_flow']
    C = dict_vars['C']
    R = dict_vars['R']
    W = dict_vars['W']

data_test = TestDataWrapper(train_ni_idx, train_cg_idx, normalize_flow)

img_map_wrapper = ENetWrapper()
flow_wrapper = NICEWrapper()
if cuda_state:
    img_map_wrapper.model.cuda()
    flow_wrapper.model.cuda()

checkpoint = torch.load(os.path.join(results_path, f'img_map_checkpoint_{EPOCHS_IMG_MAP}.pth'))
img_map_wrapper.model.load_state_dict(checkpoint['state_dict'])

checkpoint = torch.load(os.path.join(results_path, f'flow_checkpoint_{EPOCHS_FLOW}.pth'))
flow_wrapper.model.load_state_dict(checkpoint['state_dict'])

################################################################################

flow_wrapper.model.eval()

class_id = 0

for dataloader in data_test.list_dataloaders_test:

    real_labels = []
    pred_labels = []

    with torch.no_grad():
        print('Processing class ', class_id)
        for batch_idx, (data, target, idxes) in tqdm(enumerate(dataloader)):

            if cuda_state:
                data = data.cuda()

            mapping = img_map_wrapper.model.forward(data, False)
            output = flow_wrapper.model.forward(mapping)
            dist = torch.sum((output * (W ** (1 / INPUT_DIM)) - C) ** 2, dim=1)
            scores = dist - R ** 2
            results = [0 if score < 0 else 1 for score in scores]

            if class_id != 0:
                target = target / class_id

            real_labels.extend(target.data.numpy())
            pred_labels.extend(results)

    print('Accuracy: ', accuracy_score(real_labels, pred_labels))
    print('F1_score: ', f1_score(real_labels, pred_labels))

    class_id += 1
