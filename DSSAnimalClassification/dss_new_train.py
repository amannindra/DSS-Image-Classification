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


def get_ram_usage():
    """Get the process info for the current Python script"""
    process = psutil.Process(os.getpid())
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
            done_image = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)

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


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

def train_epoch(
    model, dataloader, criterion, optimizer, device, scaler, class_names, num_classes=8, epoch=0, total_epochs=1
    
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []
    
    num_batches = len(dataloader)
    print(f"\n{'─'*60}")
    print(f"🚂 TRAINING EPOCH {epoch+1}/{total_epochs}")
    print(f"{'─'*60}")
    print(f"   Total batches: {num_batches}")
    print(f"   Batch size: {dataloader.batch_size}")
    print(f"   Total samples: ~{num_batches * dataloader.batch_size}")
    
    pbar = tqdm(
        dataloader, 
        desc=f"Train E{epoch+1}", 
        unit="batch",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    for batch_idx, (images, labels, ids) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.detach().cpu().numpy().tolist())

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.detach().cpu().numpy().tolist())
        all_ids.extend(ids)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        
        # Update progress bar with current metrics
        current_loss = running_loss / total
        current_acc = correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}'
        })
        
        # Print detailed info every 25% of batches
        if (batch_idx + 1) % max(1, num_batches // 4) == 0:
            print(f"\n   📍 Batch {batch_idx+1}/{num_batches}: Loss={current_loss:.4f}, Acc={current_acc:.4f}")

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)


    misclassified_list = []

    for i, id in enumerate(all_ids):
        if all_preds[i] != all_labels[i]:
            misclassified_list.append({
                "id": id,
                "true_label": class_names[all_labels[i]],
                "predicted_label": class_names[all_preds[i]],
                "probability": all_probs[i][all_preds[i]],
            })

    misclassified_images = pd.DataFrame(misclassified_list)
    
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
    
    print(f"\n   ✅ Training Complete!")
    print(f"   📊 Final Train Metrics:")
    print(f"      Loss: {report['loss']:.4f}")
    print(f"      Accuracy: {report['acc']:.4f} ({correct}/{total})")
    print(f"      Top-3 Accuracy: {report['top3_accuracy']:.4f}")
    print(f"      Log Loss: {report['log_loss']:.4f}")
    print(f"      Macro F1: {report['macro_f1']:.4f}")
    print(f"      Misclassified: {len(report['misclassified_images'])} samples")
    
    return epoch_acc, report


def validate_epoch(model, dataloader, criterion, device, class_names, epoch=0, total_epochs=1):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []
    
    num_batches = len(dataloader)
    print(f"\n{'─'*60}")
    print(f"🔍 VALIDATION EPOCH {epoch+1}/{total_epochs}")
    print(f"{'─'*60}")
    print(f"   Total batches: {num_batches}")

    pbar = tqdm(
        dataloader, 
        desc=f"Val E{epoch+1}", 
        unit="batch",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    with torch.no_grad():
        for batch_idx, (images, labels, ids) in enumerate(pbar):
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
            
            current_loss = running_loss / total
            current_acc = correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })

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
    
    print(f"\n   ✅ Validation Complete!")
    print(f"   📊 Final Validation Metrics:")
    print(f"      Loss: {report['loss']:.4f}")
    print(f"      Accuracy: {report['acc']:.4f} ({correct}/{total})")
    print(f"      Top-3 Accuracy: {report['top3_accuracy']:.4f}")
    print(f"      Log Loss: {report['log_loss']:.4f}")
    print(f"      Macro F1: {report['macro_f1']:.4f}")
    print(f"      Misclassified: {len(report['misclassified_images'])} samples")
    
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


def model_compose():
    """Return model configurations (not instantiated models) to allow fresh creation each time"""
    all_models = {
        "model_convnextv2_huge.fcmae_ft_in22k_in1k_384": {
            "img_size": 384,
            "timm_name": "convnextv2_huge.fcmae_ft_in22k_in1k_384",
        },
        "model_convnext_xxlarge.clip_laion2b_soup_ft_in1k": {
            "img_size": 224,
            "timm_name": "convnext_xxlarge.clip_laion2b_soup_ft_in1k",
        },
        "model_convnext_large.fb_in22k_ft_in1k": {
            "img_size": 224,
            "timm_name": "convnext_large.fb_in22k_ft_in1k",
        },
    }
    return all_models


def create_model(timm_name, num_classes, device):
    """Create a fresh model with pretrained weights"""
    print(f"\n   🔄 Creating fresh model: {timm_name}")
    model = timm.create_model(
        timm_name,
        pretrained=True,
        num_classes=num_classes,
    )
    model = model.to(device)
    print(f"   ✅ Model loaded with pretrained weights")
    return model


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
                A.RandomRotate90(p=0.2),
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
        "transform_3": A.Compose(
            [
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=0.1),
                        A.GridDistortion(distort_limit=0.1),
                    ],
                    p=0.2,
                ),
                A.OneOf([
                    A.GaussianBlur(p=0.7),
                    A.MedianBlur(p=0.7),
                    A.MotionBlur(p=0.7),
                ], p=0.5),
                A.InvertImg(p=0.2),
                A.Posterize(p=0.2),
                A.CLAHE(p=0.2),
                A.PlasmaShadow(
                    shadow_intensity_range=(0.3, 0.5),
                    plasma_size=img_size,
                    roughness=3,
                    p=0.2 
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
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

    print(f"\n📁 Directory Configuration:")
    print(f"   Environment: {'🚀 SageMaker' if is_sagemaker else '💻 Local'}")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Model directory: {args.model_dir}")

    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Validate CUDA availability if requested
    if args.use_cuda and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
    elif args.use_cuda and torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

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

    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Initial RAM usage: {get_ram_usage():.2f} MB")

    if torch.cuda.is_available():
        gpu_alloc, gpu_reserved = get_gpu_memory()
        print(
            f"Initial GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB"
        )

    base_path = args.data_dir
    print(f"Base path: {base_path}")

    train_folder = os.path.join(base_path, "train_features")
    test_folder = os.path.join(base_path, "test_features")
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

    if len(dataframe) == 0:
        raise ValueError("Training dataframe is empty!")

    print(f"After loading CSV RAM usage: {get_ram_usage():.2f} MB")
    if torch.cuda.is_available():
        gpu_alloc, gpu_reserved = get_gpu_memory()
        print(
            f"After loading CSV GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB"
        )

    print(f"Loaded {len(dataframe)} training samples")
    print(f"Dataframe Columns: {list(dataframe.columns)}")
    print(f"Dataframe shape: {dataframe.shape}")

    train_df, val_df = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=42,
        stratify=np.argmax(dataframe[class_names].values, axis=1),
    )

    # Calculate class weights for handling class imbalance
    train_labels = np.argmax(train_df[class_names].values, axis=1)
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total_samples = len(train_labels)

    # Inverse frequency weighting: weight = total_samples / (num_classes * class_count)
    class_weights = total_samples / (num_classes * class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("class_weights: ", class_weights)    
    print(f"\n📊 Class Distribution (Training Set):")
    for i, name in enumerate(class_names):
        print(f"   {name:20s}: {class_counts[i]:5d} samples, weight: {class_weights[i].item():.4f}")

    batch_size = args.batch_size
    print(f"Batch size: {batch_size}")

    if torch.cuda.is_available():
        gpu_alloc, gpu_reserved = get_gpu_memory()
        print(
            f"After creating datasets GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB"
        )

    best_val_acc = {}

    models_dict = model_compose()
    for model_name, model_config in models_dict.items():
        current_img_size = model_config["img_size"]
        timm_name = model_config["timm_name"]

        print(f"\n{'='*60}")
        print(f"🤖 MODEL CONFIG: {model_name}")
        print(f"{'='*60}")
        print(f"   Image size: {current_img_size}x{current_img_size}")
        print(f"   TIMM model: {timm_name}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Create transforms for this model's image size
        augmentation_transform = transforms_compose(current_img_size)
        print(f"   Transforms: {list(augmentation_transform.keys())}")

        num_transforms = len(augmentation_transform)
        for transform_idx, (transform_key, transform_each) in enumerate(augmentation_transform.items()):
            print(f"\n{'▓'*60}")
            print(f"📦 TRANSFORM {transform_idx+1}/{num_transforms}: {transform_key}")
            print(f"{'▓'*60}")

            print(f"\n🔄 Creating fresh model (pretrained weights)...")
            model = create_model(timm_name, num_classes, device)
            model.train()
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            print(f"\n🔧 Creating datasets...")
            train_dataset = AnimalDatasetConvnext(
                train_df,
                transform=transform_each,
                folder=train_folder,
                img_size=current_img_size,
            )
            val_dataset = AnimalDatasetConvnext(
                val_df,
                transform=get_test_transform(current_img_size),
                folder=train_folder,
                img_size=current_img_size,
            )

            print(f"   Train dataset size: {len(train_dataset)}")
            print(f"   Val dataset size: {len(val_dataset)}")

            print(f"\n🔧 Creating dataloaders...")
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

            print(f"   Train batches: {len(train_loader)}")
            print(f"   Val batches: {len(val_loader)}")
            print(f"   Batch size: {batch_size}")
            print(f"   Num workers: {args.num_cpu}")
            print(f"   RAM usage: {get_ram_usage():.2f} MB")

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

            # Reset optimizer and scheduler for each transform
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            scaler = GradScaler()
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs)

            print(f"\n🎯 Starting training loop: {args.epochs} epochs")
            print(f"   Initial LR: {args.learning_rate}")
            print(f"   Scheduler: CosineAnnealingWarmRestarts (T_0={args.epochs})")

            for epoch in range(args.epochs):
                print(f"\n{'═'*60}")
                print(f"🔄 EPOCH {epoch+1}/{args.epochs} | Model: {model_name[:30]}... | Transform: {transform_key}")
                print(f"{'═'*60}")

                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Current Learning Rate: {current_lr:.8f}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gpu_alloc, gpu_reserved = get_gpu_memory()
                    print(f"   GPU Memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")
                print(f"   RAM Usage: {get_ram_usage():.2f} MB")

                train_acc, train_report = train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    scaler,
                    class_names,
                    num_classes=num_classes,
                    epoch=epoch,
                    total_epochs=args.epochs,
                )
               

                train_report["epoch"] = epoch
                train_logger.log_report(train_report)
                
                

                if torch.cuda.is_available():
                    gpu_alloc, gpu_reserved = get_gpu_memory()
                    print(f"\n   🖥️  Post-train GPU: Allocated={gpu_alloc:.2f} MB, Reserved={gpu_reserved:.2f} MB")

                val_acc, val_report = validate_epoch(
                    model, val_loader, criterion, device, class_names=class_names,
                    epoch=epoch, total_epochs=args.epochs
                )
                val_report["epoch"] = epoch

                lr_before = scheduler.get_last_lr()[0]
                scheduler.step()
                lr_after = scheduler.get_last_lr()[0]
                val_report["lr_before"] = lr_before
                val_report["lr_after"] = lr_after

                print(f"\n   📈 LR Update: {lr_before:.8f} → {lr_after:.8f}")
                val_logger.log_report(val_report)

                if torch.cuda.is_available():   
                    gpu_alloc, gpu_reserved = get_gpu_memory()
                    print(f"   🖥️  Post-val GPU: Allocated={gpu_alloc:.2f} MB, Reserved={gpu_reserved:.2f} MB")

                # Epoch summary
                print(f"\n   {'─'*50}")
                print(f"   📋 EPOCH {epoch+1} SUMMARY:")
                print(f"      Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                print(f"      Train Loss: {train_report['loss']:.4f} | Val Loss: {val_report['loss']:.4f}")
                print(f"      Best Val Acc so far: {best_val:.4f}")
                print(f"   {'─'*50}")

                if val_acc > best_val:
                    from_epoch = epoch
                    best_val = val_acc
                    print(f"\n   🏆 NEW BEST! Val Accuracy: {val_acc:.4f} (Epoch {epoch+1})")
                    print(f"   💾 Saving best model...")
                    save_model(model, models_dir, model_name, transform_key, epoch)

            # Training complete for this transform
            print(f"\n{'▓'*60}")
            print(f"✅ TRAINING COMPLETE for {transform_key}")
            print(f"{'▓'*60}")
            print(f"   Best validation accuracy: {best_val:.4f} (Epoch {from_epoch+1})")

            # Save final metrics and model
            print(f"\n💾 Saving final metrics and model...")
            final_epoch_marker = args.epochs + 100
            train_logger.save_data_metrics(final_epoch_marker)
            val_logger.save_data_metrics(final_epoch_marker)
            save_model(model, models_dir, model_name, transform_key, final_epoch_marker)

            # Run test inference
            print(f"\n🧪 Running test inference...")
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
            print(f"✅ Test inference complete!")

            # Clean up model to free GPU memory before next transform
            print(f"\n🧹 Cleaning up model and freeing memory...")
            del model
            del optimizer
            del scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"   RAM usage after cleanup: {get_ram_usage():.2f} MB")
            if torch.cuda.is_available():
                gpu_alloc, gpu_reserved = get_gpu_memory()
                print(f"   GPU after cleanup: Allocated={gpu_alloc:.2f} MB, Reserved={gpu_reserved:.2f} MB")

    print("\n✓ Training complete!")
