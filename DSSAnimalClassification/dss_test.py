import argparse
import os
from typing import Hashable
from pandas.core.series import Series
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
# import pickle
from io import BytesIO
import boto3
import gc

from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     precision_score,
#     recall_score,
#     f1_score,
# # )
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.impute import    KNNImputer
# from sklearn.linear_model import LinearRegression
# from sklearn import tree
# from sklearn.svm import SVC

import torch

import torchvision

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

import torchvision.models as models
import os
import psutil

# import cv2

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns
import os

from PIL import Image
from tqdm import tqdm
