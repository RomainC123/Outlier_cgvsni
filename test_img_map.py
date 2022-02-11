import os
import argparse
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from dataset import TestImgMapDataWrapper
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
    data_train = pickle.load(f)['data_train']

data_test = TestImgMapDataWrapper(data_train.train_ni_idx, data_train.train_cg_idx, data_train.normalize_img_map)

img_map_wrapper = ENetWrapper()
if cuda_state:
    img_map_wrapper.model.cuda()

checkpoint = torch.load(os.path.join(results_path, f'img_map_checkpoint_{EPOCHS_IMG_MAP}.pth'))
img_map_wrapper.model.load_state_dict(checkpoint['state_dict'])

################################################################################

print(data_test.df_test)

img_map_wrapper.model.eval()

real_labels = []
pred_labels = []

with torch.no_grad():

    for batch_idx, (data, target, idxes) in tqdm(enumerate(data_test.dataloader_test)):

        if cuda_state:
            data = data.cuda()
            target = target.cuda()

        output = img_map_wrapper.model.forward(data, True)
        results = torch.nn.functional.softmax(output, dim=1)
        pred = results.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        real_labels.extend(target.cpu().data.numpy())
        pred_labels.extend(pred.cpu().data.numpy())

print(accuracy_score(real_labels, pred_labels))
