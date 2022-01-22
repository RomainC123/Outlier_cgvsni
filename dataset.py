import os
import pandas as pd
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

from params.constants import NB_IMGS_TRAIN, NB_IMGS_TEST_PERCLASS, NB_CLASSES, BATCH_SIZE
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
        df_data = pd.read_csv(os.path.join(DATASET_PATH, 'dataset.csv'))  # We only keep natural images for training
        df_data = df_data.loc[df_data['Label'] == 0]

        df_train, _ = train_test_split(df_data, train_size=int(NB_IMGS_TRAIN))
        df_train = df_train.reset_index(drop=True)

        # Compunting the mean/std of each channel for training images
        list_img = []
        for row in df_train.values:
            img_path = os.path.join(DATASET_PATH, row[0])
            with open(img_path, 'rb') as f:
                list_img.append(torch.reshape(TRANSFORMS_TRAIN(Image.open(f).convert('RGB')), (3, -1)))
        full_img = torch.cat(list_img, dim=1)
        means = torch.mean(full_img, dim=1)
        stds = torch.std(full_img, dim=1)
        self.normalize = transforms.Normalize(means, stds)

        # Creating dataset and dataloader objects
        dataset_train = ImgDataset(df_train, self.normalize)
        self.dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    def get_info(self):

        info = f'Number of train samples: {NB_IMGS_TRAIN}\n'
        info += f'Batch size: {BATCH_SIZE}\n'

        return info


class TestDataWrapper:

    def __init__(self, normalize):

        self.normalize = normalize

        # Importing data and grabbing samples for our training set
        df_data = pd.read_csv(os.path.join(DATASET_PATH, 'dataset.csv'))  # We only keep natural images for training

        df_data_classes = []
        for class_id in range(NB_CLASSES):
            df_data_classes.append(df_data.loc[df_data['Label'] == class_id].sample(NB_IMGS_TEST_PERCLASS))

        df_test = pd.concat(df_data_classes, axis=0).reset_index(drop=True)

        # Creating dataset and dataloader objects
        dataset_test = ImgDataset(df_test, self.normalize)
        self.dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
