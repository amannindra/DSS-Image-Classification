import argparse
import os
import copy
import json
import gc

import pandas as pd
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.ops import StochasticDepth
from torch.cuda.amp import GradScaler, autocast
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.transforms as transforms

import timm
import psutil
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    log_loss,
)

# from dss_util import get_ram_usage, get_gpu_memory


class AnimalDatasetConvnext(Dataset):
    """Dataset for training with Albumentations transforms (expects image= keyword arg)"""

    def __init__(self, dataframe, transform=None, folder="", img_size=224):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size
        self.folder = folder
        self.label_columns = [
            "antelope_duiker",
            "bird",
            "blank",
            "civet_genet",
            "hog",
            "leopard",
            "monkey_prosimian",
            "rodent",
        ]
        self.labels = dataframe[self.label_columns].values.argmax(axis=1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]["id"]
        filename = img_id + ".jpg"
        image_path = os.path.join(self.folder, filename)

        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"cv2.imread returned None for: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if self.transform:
            transformed_image = self.transform(image=image)
            done_image = transformed_image["image"]
        else:
            done_image = torch.zeros(
                (3, self.img_size, self.img_size), dtype=torch.float32
            )

        return done_image, self.labels[idx], img_id

    def get_y(self):
        return self.labels


class AnimalDatasetTest(Dataset):
    """Dataset for testing - returns only (image, id) for inference"""

    def __init__(self, dataframe, transform=None, folder="", img_size=224):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size
        self.folder = folder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        id = self.dataframe.iloc[idx]["id"]
        filename = id + ".jpg"
        image_path = os.path.join(self.folder, filename)

        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"cv2.imread returned None for: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if self.transform:
            transformed_image = self.transform(image=image)
            done_image = transformed_image["image"]
        else:
            done_image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        return done_image, id
