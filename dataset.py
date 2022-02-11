import os
import pandas as pd
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

from params.constants import *
from params.paths import DATASET_PATH
from params.transforms import TRANSFORMS_TRAIN

################################################################################


class ImgDataset(Dataset):
    """
    Dataset class tailored for our problem
    Allows us to only load images when they are required for training
    """

    def __init__(self, df_data, transform_normalize):

        self.df_data = df_data
        self.normalize = transform_normalize

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):

        if idx > len(self.df_data):
            raise ValueError(f'Index out of bounds: {idx}')

        img_path = os.path.join(DATASET_PATH, self.df_data['Name'][idx])
        target = self.df_data['Label'][idx]

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if TRANSFORMS_TRAIN is not None:
            img = TRANSFORMS_TRAIN(img)
            img = self.normalize(img)

        return img, target, idx

################################################################################


class TrainDataWrapper:
    """
    Wrapper that processes the dataset to create a training set
    Contains the dataloader
    """

    def __init__(self):

        # Importing data and grabbing samples for our training set
        df_data = pd.read_csv(os.path.join(DATASET_PATH, 'dataset.csv'))
        df_ni = df_data.loc[df_data['Label'] == 0]
        df_cg = df_data.loc[df_data['Label'] == ID_CG_TRAIN]  # Only one type of CG image for training

        df_train_ni = df_ni.sample(NB_IMGS_TRAIN_NI)
        df_train_cg = df_cg.sample(NB_IMGS_TRAIN_CG)

        self.train_ni_idx = df_train_ni.index
        self.train_cg_idx = df_train_cg.index

        df_train_ni = df_train_ni.reset_index(drop=True)
        df_train_cg = df_train_cg.reset_index(drop=True)
        df_train = pd.concat([df_train_ni, df_train_cg], ignore_index=True)

        # Compunting the mean/std of each channel for training images for the first part (all images)
        list_img_ni = []
        list_img = []

        for row in df_train.values:
            img_path = os.path.join(DATASET_PATH, row[0])
            with open(img_path, 'rb') as f:
                img = torch.reshape(TRANSFORMS_TRAIN(Image.open(f).convert('RGB')), (3, -1))
            if row in df_ni.values:
                list_img_ni.append(img)
            list_img.append(img)

        full_img = torch.cat(list_img, dim=1)
        means = torch.mean(full_img, dim=1)
        stds = torch.std(full_img, dim=1)
        self.normalize_img_map = transforms.Normalize(means, stds)

        full_img_ni = torch.cat(list_img_ni, dim=1)
        means_ni = torch.mean(full_img_ni, dim=1)
        stds_ni = torch.std(full_img_ni, dim=1)
        self.normalize_flow = transforms.Normalize(means_ni, stds_ni)

        # Creating dataset and dataloader objects
        dataset_train_img_map = ImgDataset(df_train, self.normalize_img_map)
        dataset_train_flow = ImgDataset(df_train_ni, self.normalize_flow)

        self.dataloader_train_img_map = DataLoader(dataset_train_img_map, batch_size=BATCH_SIZE, shuffle=True)
        self.dataloader_train_flow = DataLoader(dataset_train_flow, batch_size=BATCH_SIZE, shuffle=True)

    def get_info(self):

        info = f'Number of NI train samples: {NB_IMGS_TRAIN_NI}\n'
        info += f'Number of CG train samples: {NB_IMGS_TRAIN_CG}\n'
        info += f'CG images ID: {ID_CG_TRAIN}\n'
        info += f'Batch size: {BATCH_SIZE}\n'

        return info


class TestDataWrapper:

    def __init__(self, train_ni_idx, train_cg_idx, normalize):

        self.normalize = normalize

        # Importing data, removing samples already used for training, and then sampling from each class
        df_data = pd.read_csv(os.path.join(DATASET_PATH, 'dataset.csv'))
        df_data = df_data.drop(train_ni_idx)
        df_data = df_data.drop(train_cg_idx)

        self.list_dataloaders_test = []
        for class_id in range(NB_CLASSES):
            df_class = df_data.loc[df_data['Label'] == class_id].sample(NB_IMGS_TEST_PERCLASS).reset_index(drop=True)
            dataset_class = ImgDataset(df_class, self.normalize)
            self.list_dataloaders_test.append(DataLoader(dataset_class, batch_size=BATCH_SIZE, shuffle=True))


class TestImgMapDataWrapper:

    def __init__(self, train_ni_idx, train_cg_idx, normalize):

        self.normalize = normalize

        # Importing data, removing samples already used for training, and then sampling from each class
        df_data = pd.read_csv(os.path.join(DATASET_PATH, 'dataset.csv'))
        df_data = df_data.drop(train_ni_idx)
        df_data = df_data.drop(train_cg_idx)

        list_df_class = []
        for class_id in [0, ID_CG_TRAIN]:
            list_df_class.append(df_data.loc[df_data['Label'] == class_id].sample(NB_IMGS_TEST_PERCLASS).reset_index(drop=True))

        self.df_test = pd.concat(list_df_class, axis=0).reset_index(drop=True)
        self.dataloader_test = DataLoader(ImgDataset(self.df_test, self.normalize), batch_size=BATCH_SIZE, shuffle=True)
