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
from torchvision.ops import StochasticDepth
from torch.cuda.amp import GradScaler, autocast
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.transforms as transforms

import timm
import psutil
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    hinge_loss)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn import tree 
from sklearn.svm import SVC


from torchvision.models import resnet18, ResNet18_Weights, swin_b
import torch

import torchvision
from torchvision.models import Swin_T_Weights, ResNet18_Weights, Swin_S_Weights, Swin_B_Weights
import torchvision.models as models
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights, ResNet50_Weights

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


def get_ram_usage():
    # Get the process info for the current Python script
    process = psutil.Process(os.getpid())
    # Return the Resident Set Size (RSS) in megabytes
    return process.memory_info().rss / (1024 * 1024)

def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        return allocated, reserved
    return 0, 0


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
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
      
        
          # train_transform = transforms.Compose(
    #     [
    #         transforms.RandomRotation(degrees=(0, 30), fill=128),
    #         transforms.Resize((img_size, img_size)),
    #         transforms.RandomHorizontalFlip(p=0.4),
    #         transforms.ColorJitter(brightness=(0.7, 1.2), contrast=(0.7, 1.2), saturation=(0.7, 1.2)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         # transforms.RandomErasing(),
    #     ]
    # )
        antelope_transform = transforms.Compose([
          transforms.Resize((img_size, img_size)),
          transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        bird_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        blank_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(0, 30), fill=128),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ColorJitter(brightness=(0.7, 1.2), contrast=(0.7, 1.2), saturation=(0.7, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        civet_genet_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        hog_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        leopard_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(0, 30), fill=128),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ColorJitter(brightness=(0.7, 1.2), contrast=(0.7, 1.2), saturation=(0.7, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        monkey_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        rodent_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        
        self.class_transforms = {
            "blank": blank_transform,
            "antelope_duiker": antelope_transform,
            "rodent": rodent_transform,
            "monkey_prosimian": monkey_transform,
            "bird": bird_transform,
            "hog": hog_transform,
            "leopard": leopard_transform,
            "civet_genet": civet_genet_transform,
        }
        self.labels = dataframe[self.label_columns].values.argmax(axis=1)
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

        return done_image, self.labels[idx], self.dataframe.iloc[idx]["id"]

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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def train_epoch(model, dataloader, criterion, optimizer, device, class_names, num_classes=8):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # For storing prediction probabilities
    all_ids = []
    
    # Initialize Scaler for Mixed Precision
    
    # pbar = tqdm(dataloader, desc="Training")
    print("Training...")
    pbar = tqdm(dataloader, desc="Training")
    for images, labels, ids in pbar:
        images, labels = images.to(device), labels.to(device)
        # ids are strings, don't move to device

        optimizer.zero_grad()

        # 1. Cast forward pass to float16 (Auto Mixed Precision)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item() * images.size(0)
        
        # Get probabilities for additional metrics
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.detach().cpu().numpy().tolist())
        all_ids.extend(ids)  # ids are already strings, not tensors
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        # pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")
    
    # Convert to numpy arrays for easier manipulation
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    misclassified_images = pd.DataFrame(
        columns=["id", "true_label", "predicted_label", "probability"]
    )
    for i, id in enumerate(all_ids):
        if all_preds[i] != all_labels[i]:
            misclassified_images.loc[len(misclassified_images)] = {
                "id": id,
                "true_label": class_names[all_labels[i]],
                "predicted_label": class_names[all_preds[i]],
                "probability": all_probs[i][all_preds[i]]
            }

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
    report["loss"] = float(running_loss / total)
    report["acc"] = float(correct / total)
    report["ids"] = all_ids
    epoch_acc = float(correct / total)
    report["misclassified_images"] = misclassified_images

    report["log_loss"] = float(log_loss(all_labels, all_probs))

    top3_correct = 0
    for i, label in enumerate(all_labels):
        top3_preds = np.argsort(all_probs[i])[-3:]
        if label in top3_preds:
            top3_correct += 1
    report["top3_accuracy"] = float(top3_correct / len(all_labels))

    class_confidences = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_preds == i
        if class_mask.sum() > 0:
            class_confidences[class_name] = float(all_probs[class_mask, i].mean())
        else:
            class_confidences[class_name] = 0.0
    report['class_confidences'] = class_confidences
    
    # 4. Ensure macro/weighted metrics are present (sklearn returns np.float64)
    report['macro_precision'] = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    report['macro_recall'] = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    report['macro_f1'] = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    
    report['micro_precision'] = float(precision_score(all_labels, all_preds, average='micro', zero_division=0))
    report['micro_recall'] = float(recall_score(all_labels, all_preds, average='micro', zero_division=0))
    report['micro_f1'] = float(f1_score(all_labels, all_preds, average='micro', zero_division=0))
    
    report['weighted_precision'] = float(precision_score(all_labels, all_preds, average='weighted', zero_division=0))
    report['weighted_recall'] = float(recall_score(all_labels, all_preds, average='weighted', zero_division=0))
    report['weighted_f1'] = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))
    
    # Verify required keys exist
    assert 'macro avg' in report, "macro avg missing from classification_report"
    assert 'weighted avg' in report, "weighted avg missing from classification_report"
    assert 'accuracy' in report, "accuracy missing from classification_report"
    assert 'micro_precision' in report, "micro_precision missing from classification_report"
    assert 'micro_recall' in report, "micro_recall missing from classification_report"
    assert 'micro_f1' in report, "micro_f1 missing from classification_report"
    assert 'weighted_precision' in report, "weighted_precision missing from classification_report"
    assert 'weighted_recall' in report, "weighted_recall missing from classification_report"
    assert 'weighted_f1' in report, "weighted_f1 missing from classification_report"
    
    print(f"Train Acc: {report['acc']:.4f}, Top-3 Acc: {report['top3_accuracy']:.4f}, Log Loss: {report['log_loss']:.4f}")
    return epoch_acc, report

def validate_epoch(model, dataloader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []

    pbar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for images, labels, ids in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.detach().cpu().numpy().tolist())

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_ids.extend(ids)
            all_preds.extend(predicted.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    misclassified_images = pd.DataFrame(
        columns=["id", "true_label", "predicted_label", "probability"]
    )
    for i, id in enumerate(all_ids):
        if all_preds[i] != all_labels[i]:
            misclassified_images.loc[len(misclassified_images)] = {
                "id": id,
                "true_label": class_names[all_labels[i]],
                "predicted_label": class_names[all_preds[i]],
                "probability": all_probs[i][all_preds[i]],
            }

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
    report["loss"] = float(running_loss / total)
    report["acc"] = float(correct / total)
    report["ids"] = all_ids
    epoch_acc = float(correct / total)
    report["misclassified_images"] = misclassified_images

    report["log_loss"] = float(log_loss(all_labels, all_probs))

    top3_correct = 0
    for i, label in enumerate(all_labels):
        top3_preds = np.argsort(all_probs[i])[-3:]
        if label in top3_preds:
            top3_correct += 1
    report["top3_accuracy"] = float(top3_correct / len(all_labels))

    class_confidences = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_preds == i
        if class_mask.sum() > 0:
            class_confidences[class_name] = float(all_probs[class_mask, i].mean())
        else:
            class_confidences[class_name] = 0.0
    report['class_confidences'] = class_confidences
    
    # 4. Ensure macro/weighted metrics are present (sklearn returns np.float64)
    report['macro_precision'] = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    report['macro_recall'] = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    report['macro_f1'] = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    
    report['micro_precision'] = float(precision_score(all_labels, all_preds, average='micro', zero_division=0))
    report['micro_recall'] = float(recall_score(all_labels, all_preds, average='micro', zero_division=0))
    report['micro_f1'] = float(f1_score(all_labels, all_preds, average='micro', zero_division=0))
    
    report['weighted_precision'] = float(precision_score(all_labels, all_preds, average='weighted', zero_division=0))
    report['weighted_recall'] = float(recall_score(all_labels, all_preds, average='weighted', zero_division=0))
    report['weighted_f1'] = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))
    
    # Verify required keys exist
    assert 'macro avg' in report, "macro avg missing from classification_report"
    assert 'weighted avg' in report, "weighted avg missing from classification_report"
    assert 'accuracy' in report, "accuracy missing from classification_report"
    assert 'micro_precision' in report, "micro_precision missing from classification_report"
    assert 'micro_recall' in report, "micro_recall missing from classification_report"
    assert 'micro_f1' in report, "micro_f1 missing from classification_report"
    assert 'weighted_precision' in report, "weighted_precision missing from classification_report"
    assert 'weighted_recall' in report, "weighted_recall missing from classification_report"
    assert 'weighted_f1' in report, "weighted_f1 missing from classification_report"
    
    print(f"Validation Loss: {report['loss']:.4f}, Validation Acc: {report['acc']:.4f}, Top-3 Acc: {report['top3_accuracy']:.4f}, Log Loss: {report['log_loss']:.4f}")
    return epoch_acc, report



class TrainingLogger:
    """Professional logging for SageMaker training"""

    def __init__(
        self,
        data_output_dir,
        name="metrics",
        class_names=None,
        transform_name=None,
    ):
        self.data_output_dir = data_output_dir
        os.makedirs(self.data_output_dir, exist_ok=True)
        self.history = []
        self.class_names = class_names if class_names is not None else []
        self.name = name
        self.transform_name = transform_name

    def log_report(self, report):
        """Log report for one epoch"""
        class_f1s_dict = {
            class_name: report[class_name]["f1-score"]
            for class_name in self.class_names
        }
        class_precisions_dict = {
            class_name: report[class_name]["precision"]
            for class_name in self.class_names
        }
        class_recalls_dict = {
            class_name: report[class_name]["recall"] for class_name in self.class_names
        }
        report["class_f1s"] = class_f1s_dict
        report["class_precisions"] = class_precisions_dict
        report["class_recalls"] = class_recalls_dict
        self.history.append(report)
        # print(f"Report: {report}")  # Too verbose - removed
        # self.print_log(report)
        
    def print_log(self, report):
        print(
            f"\n[METRICS] epoch={report['epoch']} {self.name}_loss={report['loss']:.4f} "
            f"{self.name}_acc={report['acc']:.4f}"
        )

        print(f"  Top-3 Accuracy: {report.get('top3_accuracy', 0):.4f}")
        print(f"  Log Loss: {report.get('log_loss', 0):.4f}")

        if "macro avg" in report:
            macro_avg = report["macro avg"]
            print(
                f"  Macro Avg    - F1: {macro_avg['f1-score']:.4f}, "
                f"Precision: {macro_avg['precision']:.4f}, Recall: {macro_avg['recall']:.4f}"
            )

        if "weighted avg" in report:
            weighted_avg = report["weighted avg"]
            print(
                f"  Weighted Avg - F1: {weighted_avg['f1-score']:.4f}, "
                f"Precision: {weighted_avg['precision']:.4f}, Recall: {weighted_avg['recall']:.4f}"
            )

        if "class_f1s" in report and self.class_names:
            print("\n  Per-Class F1 Scores:")
            for class_name in self.class_names:
                f1 = report.get("class_f1s", {}).get(class_name, 0)
                conf = report.get("class_confidences", {}).get(class_name, 0)
                print(f"    {class_name:20s}: F1={f1:.4f}, Confidence={conf:.4f}")

    def save_data_metrics(self, epoch):
        """Save complete history to JSON with proper handling of numpy types"""

        print(f"Saving metrics to {self.data_output_dir}")

        clean_history = []
        for report in self.history:
            clean_report = copy.deepcopy(report)

            if "ids" in clean_report:
                del clean_report["ids"]

            if "misclassified_images" in clean_report:
                if hasattr(clean_report["misclassified_images"], "to_dict"):
                    clean_report["misclassified_images"] = clean_report[
                        "misclassified_images"
                    ].to_dict("records")

            clean_history.append(clean_report)

        json_path = os.path.join(
            self.data_output_dir, f"{self.name}_metrics_{epoch}.json"
        )
        with open(json_path, "w") as f:
            json.dump(clean_history, f, indent=2, cls=NumpyEncoder)

        if clean_history:
            keys = list(clean_history[0].keys())
            df = pd.DataFrame(columns=keys)
            for report in clean_history:
                row = pd.DataFrame([report])
                df = pd.concat([df, row], ignore_index=True)
            df.to_csv(
                os.path.join(self.data_output_dir, f"{self.name}_metrics_{epoch}.csv"),
                index=False,
            )

        print(f"✓ Metrics saved to {self.data_output_dir}")
        print(f"  - {json_path}")
        print(f"  Total epochs logged: {len(clean_history)}")


def model_compose(num_classes):
    all_models = {
        "model_convnextv2_huge.fcmae_ft_in22k_in1k_384": {
            "img_size": 384,
            "model": timm.create_model(
                "convnextv2_huge.fcmae_ft_in22k_in1k_384",
                pretrained=True,
                num_classes=num_classes,
            ),
        },
        "model_convnext_xxlarge.clip_laion2b_soup_ft_in1k": {
            "img_size": 224,
            "model": timm.create_model(
                "convnext_xxlarge.clip_laion2b_soup_ft_in1k",
                pretrained=True,
                num_classes=num_classes,
            ),
        },
        "model_convnext_large.fb_in22k_ft_in1k": {
            "img_size": 224,
            "model": timm.create_model(
                "convnext_large.fb_in22k_ft_in1k",
                pretrained=True,
                num_classes=num_classes,
            ),
        },
    }
    return all_models


def transforms_compose(img_size):
    """Create augmentation transforms with the specified image size"""
    all_transforms = {
        "transform_1": A.Compose(
            [
                A.Resize(height=img_size + 20, width=img_size + 20, p=1.0),
                A.RandomCrop(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, p=0.5),
                A.Affine(
                    translate_percent=0.1, scale=(0.8, 1.2), rotate=(-30, 30), p=0.5
                ),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        ),
        "transform_2": A.Compose(
            [
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5
                ),
                A.OneOf(
                    [
                        A.GaussNoise(std_range=(0.1, 0.2)),
                        A.GaussianBlur(blur_limit=7),
                        A.MotionBlur(blur_limit=7),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=0.1),
                        A.GridDistortion(distort_limit=0.1),
                    ],
                    p=0.2,
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    p=0.3,
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        ),
    }
    return all_transforms


def get_test_transform(img_size):
    """Create test/validation transform using Albumentations"""
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def save_model(model, models_dir, model_name, key, epoch):
    """Save model to models_dir/model_name_key_model_epoch.pth"""
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}_{key}_model_{epoch}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    return model_path


def compute_test_model(
    model,
    models_dir,
    model_name,
    key,
    from_epoch,
    device,
    class_names,
    batch_size,
    num_cpu,
    test_img_size,
    test_folder,
    test_df,
    test_output_dir,
):
    """Run inference on test set and save predictions"""
    model_path = os.path.join(
        models_dir, f"{model_name}_{key}_model_{from_epoch}.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Use Albumentations transform for test (matches training pipeline)
    test_transform = get_test_transform(test_img_size)

    test_dataset = AnimalDatasetTest(
        test_df, transform=test_transform, folder=test_folder, img_size=test_img_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cpu
    )

    rows = []
    os.makedirs(test_output_dir, exist_ok=True)

    pbar = tqdm(test_loader, total=len(test_loader), desc="Testing")
    for batch_idx, (images, ids) in enumerate(pbar):
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

        probs = probs.cpu().numpy()

        for i in range(len(ids)):
            rows.append([ids[i], *probs[i]])

    submission_df = pd.DataFrame(rows, columns=["id"] + class_names)
    output_path = os.path.join(test_output_dir, f"submission_{from_epoch}.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"✓ Test predictions saved to {output_path}")


if __name__ == "__main__":
    print("Starting training...")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--epochs", type=int, default=5)
    # parser.add_argument("--batch-size", type=int, default=16)  # Reduced from 32 to 16 for Swin Transformer
    # parser.add_argument("--learning-rate", type=float, default=0.001)
    # parser.add_argument("--use-cuda", type=lambda x: (str(x).lower() == 'true'), default=True)
    # parser.add_argument("--weight-decay", type=float, default=0.01)
    # parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    # parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    # parser.add_argument("--save-file", type=str, default="final_swin_t_model.pth")
    # parser.add_argument("--num-cpu", type=int, default=4)
    # parser.add_argument("--image-size", type=int, default=224)
    # parser.add_argument("--stochastic-depth", type=float, default=0.1)
    parser = argparse.ArgumentParser(
        description='Train model for Animal Classification (Local & SageMaker)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Input image size (224 or 384)"
    )

    # System configuration
    parser.add_argument("--use-cuda", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Use CUDA if available (true/false)")
    parser.add_argument("--num-cpu", type=int, default=4,
                        help="Number of CPU workers for data loading")
    
    # Data paths - Works for both Local and SageMaker
    # SageMaker: Uses SM_CHANNEL_TRAINING env var
    # Local: Defaults to ./data
    parser.add_argument("--data-dir", type=str, 
                        default=os.environ.get("SM_CHANNEL_TRAINING", "./data"),
                        help="Directory containing training data (train_labels.csv and train_features/)")
    
    # Model directory - Works for both Local and SageMaker
    # SageMaker: Uses SM_MODEL_DIR env var (auto-uploaded to S3)
    # Local: Defaults to ./models
    parser.add_argument("--model-dir", type=str, 
                        default=os.environ.get("SM_MODEL_DIR", "./models"),
                        help="Directory to save trained models")
    
    # Output directory - Works for both Local and SageMaker
    # SageMaker: Uses SM_OUTPUT_DATA_DIR env var (metrics uploaded to S3)
    # Local: Defaults to ./output
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
                        help="Directory to save metrics and logs")
    
    # Model saving
    parser.add_argument(
        "--save-file",
        type=str,
        default="final_model.pth",
        help="Filename for final saved model",
    )

    args = parser.parse_args()
    
    print(f"Arguments: {args}")
    print(f"Initial RAM usage: {get_ram_usage():.2f} MB")
    
    # Detect running environment
    is_sagemaker = os.environ.get("SM_MODEL_DIR") is not None
    
    print(f"SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR')}")
    print(f"SM_OUTPUT_DATA_DIR: {os.environ.get('SM_OUTPUT_DATA_DIR')}")
    
    # Create directories if they don't exist (for local training)
    if not is_sagemaker:
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n📁 Directory Configuration:")
    print(f"   Environment: {'🚀 SageMaker' if is_sagemaker else '💻 Local'}")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Model directory: {args.model_dir}")
    print(f"   Output directory: {args.output_dir}")
    # os.makedirs(args.model_dir, exist_ok=True)
    # print(f"Model directory: {args.model_dir}")

    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    # Validate CUDA availability if requested
    if args.use_cuda and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
    elif args.use_cuda and torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    class_names = [
        "antelope_duiker",
        "bird",
        "blank",
        "civet_genet",
        "hog",
        "leopard",
        "monkey_prosimian",
        "rodent",
    ]
    num_classes = len(class_names)
    output_channels = 3
    
    # Image size: 224 is much more memory-efficient than 384 for Swin Transformer
    # Swin Transformer works well with 224x224 images
    img_size = args.image_size  # Changed from 384 to reduce memory usage
    
    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Output channels: {output_channels}")
    print(f"Image size: {img_size} (optimized for Swin Transformer memory usage)")
    print(f"Initial RAM usage: {get_ram_usage():.2f} MB")
    if torch.cuda.is_available():
        gpu_alloc, gpu_reserved = get_gpu_memory()
        print(
            f"Initial GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB"
        )

    base_path = args.data_dir
    print(f"Base path: {base_path}")
    
    # bucket = "animal-classification-dss-works"
    train_folder = os.path.join(base_path, "train_features")
    test_folder = os.path.join(base_path, "test_features")
    train_features_csv = os.path.join(base_path, "train_features.csv")
    test_features_csv = os.path.join(base_path, "test_features.csv")
    train_labels_csv = os.path.join(base_path, "train_labels.csv")
    
    if not os.path.exists(train_labels_csv):
        raise FileNotFoundError(f"Training labels CSV not found: {train_labels_csv}")
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"Training images folder not found: {train_folder}")
    if not os.path.exists(test_features_csv):
        print(f"Warning: Test features CSV not found: {test_features_csv}")

    dataframe = pd.read_csv(train_labels_csv)
    dataframe_test = pd.read_csv(test_features_csv)
    print(f"Dataframe train: {dataframe.head()}")
    # print(f"Dataframe test: {dataframe_test.head()}")
    
    # Validate dataframe is not empty
    if len(dataframe) == 0:
        raise ValueError("Training dataframe is empty!")
    

    print(f"After loading CSV RAM usage: {get_ram_usage():.2f} MB")
    if torch.cuda.is_available():
        gpu_alloc, gpu_reserved = get_gpu_memory()
        print(f"After loading CSV GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")
    
    
    
    print(f"DataframeLoaded {len(dataframe)} training samples")
    print(f"Dataframe Columns: {list(dataframe.columns)}")
    print(f"Dataframe sample:\n{dataframe.head()}")
    print(f"Dataframe shape: {dataframe.shape}")

    train_df, val_df = train_test_split( # type: ignore
        dataframe,
        test_size=0.25,
        random_state=42,
        stratify=np.argmax(dataframe[class_names].values, axis=1),
    )

    batch_size = args.batch_size
    print(f"Batch size: {batch_size}")

    if torch.cuda.is_available():
        gpu_alloc, gpu_reserved = get_gpu_memory()
        print(
            f"After creating datasets GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB"
        )

    best_val_acc = {}

    models_dict = model_compose(num_classes)
    for model_name, model_config in models_dict.items():
        current_img_size = model_config["img_size"]
        model = model_config["model"].to(device)
        model.train()

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Image size: {current_img_size}")
        print(f"{'='*60}")

        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Create transforms for this model's image size
        augmentation_transform = transforms_compose(current_img_size)

        for transform_key, transform_each in augmentation_transform.items():
            print(f"\n--- Transform: {transform_key} ---")

            train_dataset = AnimalDatasetConvnext(
                train_df,
                transform=transform_each,
                folder=train_folder,
                img_size=current_img_size,
            )
            val_dataset = AnimalDatasetConvnext(
                val_df,
                transform=transform_each,
                folder=train_folder,
                img_size=current_img_size,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.num_cpu,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_cpu,
            )

            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            print(f"After loading model RAM usage: {get_ram_usage():.2f} MB")

            # Directory structure: model_dir/model_name/transform_key/{models,data,test}
            base_dir = os.path.join(args.model_dir, model_name, transform_key)
            models_dir = os.path.join(base_dir, "models")
            data_dir = os.path.join(base_dir, "data")
            test_dir = os.path.join(base_dir, "test")

            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            print(f"\n📁 Output Structure:")
            print(f"   Base: {base_dir}")
            print(f"   Models: {models_dir}")
            print(f"   Data (JSON/CSV): {data_dir}")
            print(f"   Test: {test_dir}")

            train_logger = TrainingLogger(
                data_output_dir=data_dir,
                class_names=class_names,
                name=f"train",
                transform_name=transform_key,
            )
            val_logger = TrainingLogger(
                data_output_dir=data_dir,
                class_names=class_names,
                name=f"val",
                transform_name=transform_key,
            )

            from_epoch = 0
            best_val = 0.0

            for epoch in range(args.epochs):
                print(f"\nEpoch {epoch+1}/{args.epochs}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gpu_alloc, gpu_reserved = get_gpu_memory()
                    print(
                        f"Start of epoch GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB"
                    )

                train_acc, train_report = train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    num_classes=num_classes,
                    class_names=class_names,
                )
                train_report["epoch"] = epoch
                train_logger.log_report(train_report)
                train_logger.print_log(train_report)

                if torch.cuda.is_available():
                    gpu_alloc, gpu_reserved = get_gpu_memory()
                    print(
                        f"After training GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB"
                    )

                val_acc, val_report = validate_epoch(
                    model, val_loader, criterion, device, class_names=class_names
                )
                val_report["epoch"] = epoch
                lr_before = scheduler.get_last_lr()[0]
                scheduler.step()
                lr_after = scheduler.get_last_lr()[0]
                val_report["lr_before"] = lr_before
                val_report["lr_after"] = lr_after
                val_logger.log_report(val_report)
                val_logger.print_log(val_report)

                print(f"Learning Rate: {lr_before:.6f} → {lr_after:.6f}")

                if torch.cuda.is_available():
                    gpu_alloc, gpu_reserved = get_gpu_memory()
                    print(
                        f"After validation GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB"
                    )

                if val_acc > best_val:
                    from_epoch = epoch
                    best_val = val_acc
                    print(f"New best validation accuracy: {val_acc:.4f}")
                    save_model(model, models_dir, model_name, transform_key, epoch)

            # Save final metrics and model
            final_epoch_marker = args.epochs + 100
            train_logger.save_data_metrics(final_epoch_marker)
            val_logger.save_data_metrics(final_epoch_marker)
            save_model(model, models_dir, model_name, transform_key, final_epoch_marker)

            # Run test inference
            compute_test_model(
                model,
                models_dir,
                model_name,
                transform_key,
                from_epoch,
                device,
                class_names,
                batch_size,
                args.num_cpu,
                current_img_size,
                test_folder,
                dataframe_test,
                test_dir,
            )

    print("\n✓ Training complete!")
