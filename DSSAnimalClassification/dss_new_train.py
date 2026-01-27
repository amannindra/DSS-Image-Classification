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
# import pickle
# from io import BytesIO
# import boto3

import json
import gc

"""
METRICS EXPLANATION:
====================

BASIC METRICS:
- accuracy: Overall % correct predictions (good baseline, but can be misleading with imbalanced data)
- loss (cross-entropy): Training objective being minimized. Lower = better fit to training data

CONFIDENCE METRICS:
- log_loss: Measures quality of probability predictions. Lower = better calibrated confidence
  * Range: 0 (perfect) to infinity (worse)
  * Example: If model predicts 99% confidence and is right, low log_loss. If 51% and right, higher log_loss
  
- top3_accuracy: % of samples where correct class is in top 3 predictions
  * Useful when close calls are acceptable (e.g., bird vs blank, two similar species)
  * Higher = model at least narrows down possibilities well

PER-CLASS METRICS (in classification_report):
- precision: Of all predictions for class X, what % were actually X? (Avoiding false alarms)
  * High precision = when model says "leopard", it's usually right
  
- recall: Of all actual X samples, what % did we find? (Avoiding misses)
  * High recall = we catch most of the leopards in the dataset
  
- f1-score: Harmonic mean of precision and recall (balances both)
  * Use this as main per-class metric. Range: 0 to 1, higher = better

- support: Number of actual samples of this class in the dataset

AGGREGATED METRICS:
- macro avg: Simple average across all classes (treats all classes equally)
  * Use when all animal types are equally important
  
- weighted avg: Average weighted by number of samples per class
  * Use when some animals are more common/important
  
- micro avg: Global average (usually same as accuracy for multi-class)

PER-CLASS CONFIDENCE:
- Average prediction confidence when model predicts each class
- High confidence + high F1 = model is sure and correct
- High confidence + low F1 = model is overconfident and wrong (needs calibration)
- Low confidence = model is uncertain (may need more training data for that class)

CONFUSION MATRIX:
- Shows which classes get confused with each other
- Row = actual class, Column = predicted class
- Diagonal = correct predictions
"""

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


from torchvision.models import resnet18, ResNet18_Weights
import torch

import torchvision
from torchvision.models import Swin_T_Weights, ResNet18_Weights
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


class AnimalDataset(Dataset):
    def __init__(self, dataframe, transform=None, folder="", img_size=224):
   
        self.dataframe = dataframe.reset_index(drop=True)
        
        self.transform = transform
        self.img_size = img_size
        # Extract labels (one-hot to class index)
        label_columns = [
            "antelope_duiker",
            "bird",
            "blank",
            "civet_genet",
            "hog",
            "leopard",
            "monkey_prosimian",
            "rodent",
        ]
        y = dataframe[label_columns].values
        assert (y.sum(axis=1) == 1).all()
        self.labels = y.argmax(axis=1).astype("int64")
        self.folder = folder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image (numpy array) from DataFrame
        id = self.dataframe.iloc[idx]["id"]
        filename = id + ".jpg"
        image_path = os.path.join(self.folder, filename)
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Log error but don't print for every missing image to avoid spam
            if idx < 10:  # Only print first 10 errors
                print(f"Error loading {image_path}: {e}")
            # Create a blank image as fallback (will hurt training but prevents crash)
            image = Image.new('RGB', (self.img_size, self.img_size), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]
    def get_y(self):
        return self.labels

    def show_image(self, idx):
        image = self.dataframe.iloc[idx]["image"]
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    def return_numpy_image(self, idx):
        return self.dataframe.iloc[idx]["image"]

    def return_transformed_image(self, idx):
        image = self.dataframe.iloc[idx]["image"]
        image = Image.fromarray(image.astype('uint8'))
        if self.transform:
            image = self.transform(image)
        return image

def train_epoch(model, dataloader, criterion, optimizer, device, class_names, num_classes=8):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # For storing prediction probabilities
    
    # Initialize Scaler for Mixed Precision
    
    # pbar = tqdm(dataloader, desc="Training")
    print("Training...")
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # 1. Cast forward pass to float16 (Auto Mixed Precision)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # with autocast():
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)

        # # 2. Scale loss and backward pass
        # if scaler is not None:
        #     scaler.scale(loss).backward()
        # else:
        #     loss.backward()
        
        # # 3. Step optimizer using scaler
        # if scaler is not None:
        #     scaler.step(optimizer)
        # else:
        #     optimizer.step()
        # if scaler is not None:
        #     scaler.update()

        # Statistics (keep existing logic)
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item() * images.size(0)
        
        # Get probabilities for additional metrics
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.detach().cpu().numpy())
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        # pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels_array = np.array(all_labels)
    all_preds_array = np.array(all_preds)
    
    # Basic metrics
    report = classification_report(all_labels_array, all_preds_array, target_names=class_names, output_dict=True)
    # Confusion matrix with numeric labels (0 to num_classes-1)
    report['confusion_matrix'] = confusion_matrix(all_labels_array, all_preds_array).tolist()
    report['loss'] = running_loss / total
    report['acc'] = correct / total
    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels_array, all_preds_array)
    
    # Additional valuable metrics
    # 1. Log Loss (measures prediction confidence quality)
    report['log_loss'] = log_loss(all_labels_array, all_probs)
    
    # 2. Top-3 Accuracy (useful for multi-class: is correct class in top 3 predictions?)
    top3_correct = 0
    for i, label in enumerate(all_labels_array):
        top3_preds = np.argsort(all_probs[i])[-3:]  # Get indices of top 3 predictions
        if label in top3_preds:
            top3_correct += 1
    report['top3_accuracy'] = top3_correct / len(all_labels_array)
    
    # 3. Per-class confidence (average confidence when predicting each class)
    class_confidences = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_preds_array == i
        if class_mask.sum() > 0:
            # Average max probability when predicting this class
            class_confidences[class_name] = float(all_probs[class_mask, i].mean())
        else:
            class_confidences[class_name] = 0.0
    report['class_confidences'] = class_confidences
    
    # 4. Macro-averaged precision, recall, f1 (already in classification_report but extract for clarity)
    report['macro_precision'] = precision_score(all_labels_array, all_preds_array, average='macro', zero_division=0)
    report['macro_recall'] = recall_score(all_labels_array, all_preds_array, average='macro', zero_division=0)
    report['macro_f1'] = f1_score(all_labels_array, all_preds_array, average='macro', zero_division=0)
    
    print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Top-3 Acc: {report['top3_accuracy']:.4f}, Log Loss: {report['log_loss']:.4f}")
    return epoch_loss, epoch_acc, report

def validate_epoch(model, dataloader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # For storing prediction probabilities

    pbar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get probabilities for additional metrics
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels_array = np.array(all_labels)
    all_preds_array = np.array(all_preds)
    
    # Basic metrics
    report = classification_report(all_labels_array, all_preds_array, target_names=class_names, output_dict=True)
    report['confusion_matrix'] = confusion_matrix(all_labels_array, all_preds_array).tolist()
    report['loss'] = running_loss / total
    report['acc'] = correct / total
    epoch_acc = correct / total
    
    # Additional valuable metrics
    # 1. Log Loss (measures prediction confidence quality)
    report['log_loss'] = log_loss(all_labels_array, all_probs)
    
    # 2. Top-3 Accuracy (useful for multi-class: is correct class in top 3 predictions?)
    top3_correct = 0
    for i, label in enumerate(all_labels_array):
        top3_preds = np.argsort(all_probs[i])[-3:]  # Get indices of top 3 predictions
        if label in top3_preds:
            top3_correct += 1
    report['top3_accuracy'] = top3_correct / len(all_labels_array)
    
    # 3. Per-class confidence (average confidence when predicting each class)
    class_confidences = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_preds_array == i
        if class_mask.sum() > 0:
            # Average max probability when predicting this class
            class_confidences[class_name] = float(all_probs[class_mask, i].mean())
        else:
            class_confidences[class_name] = 0.0
    report['class_confidences'] = class_confidences
    
    # 4. Macro-averaged precision, recall, f1 (already in classification_report but extract for clarity)
    report['macro_precision'] = precision_score(all_labels_array, all_preds_array, average='macro', zero_division=0)
    report['macro_recall'] = recall_score(all_labels_array, all_preds_array, average='macro', zero_division=0)
    report['macro_f1'] = f1_score(all_labels_array, all_preds_array, average='macro', zero_division=0)
    
    print(f"Validation Loss: {report['loss']:.4f}, Validation Acc: {report['acc']:.4f}, Top-3 Acc: {report['top3_accuracy']:.4f}, Log Loss: {report['log_loss']:.4f}")
    return epoch_acc, report

class TrainingLogger:
    """Professional logging for SageMaker training"""
    
    def __init__(self, output_dir=None, name="metrics", class_names=None):
        self.output_dir = output_dir or os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = []
        self.class_names = class_names
        self.name = name
    
    def log_report(self, report):
        """Log report for one epoch"""
        class_f1s_dict = {class_name: report[class_name]['f1-score'] for class_name in self.class_names}
        class_precisions_dict = {class_name: report[class_name]['precision'] for class_name in self.class_names}
        class_recalls_dict = {class_name: report[class_name]['recall'] for class_name in self.class_names}
        report['class_f1s'] = class_f1s_dict
        report['class_precisions'] = class_precisions_dict
        report['class_recalls'] = class_recalls_dict
        self.history.append(report)
        # print(f"Report: {report}")  # Too verbose - removed
        # self.print_log(report)
        
    def print_log(self, report):
        print(f"\n[METRICS] epoch={report['epoch']} {self.name}_loss={report['loss']:.4f} {self.name}_acc={report['acc']:.4f}")
        
        # Key metrics
        print(f"  Top-3 Accuracy: {report.get('top3_accuracy', 0):.4f}")
        print(f"  Log Loss: {report.get('log_loss', 0):.4f}")
        
        # Aggregated metrics (safely handle missing keys)
        if 'micro avg' in report:
            micro_avg = report['micro avg']
            print(f"\n  Micro Avg    - F1: {micro_avg['f1-score']:.4f}, Precision: {micro_avg['precision']:.4f}, Recall: {micro_avg['recall']:.4f}")
        
        if 'macro avg' in report:
            macro_avg = report['macro avg']
            print(f"  Macro Avg    - F1: {macro_avg['f1-score']:.4f}, Precision: {macro_avg['precision']:.4f}, Recall: {macro_avg['recall']:.4f}")
        
        if 'weighted avg' in report:
            weighted_avg = report['weighted avg']
            print(f"  Weighted Avg - F1: {weighted_avg['f1-score']:.4f}, Precision: {weighted_avg['precision']:.4f}, Recall: {weighted_avg['recall']:.4f}")
        
        # Per-class F1 scores
        if 'class_f1s' in report and self.class_names:
            print(f"\n  Per-Class F1 Scores:")
            for class_name in self.class_names:
                f1 = report.get('class_f1s', {}).get(class_name, 0)
                conf = report.get('class_confidences', {}).get(class_name, 0)
                print(f"    {class_name:20s}: F1={f1:.4f}, Confidence={conf:.4f}")
        print()
        
    
    def save(self):
        """Save complete history"""
        
        # Save as JSON (complete data)
        json_path = os.path.join(self.output_dir, f'{self.name}_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save as CSV (easy viewing)
        csv_path = os.path.join(self.output_dir, f'{self.name}_metrics.csv')
        df = pd.DataFrame(self.history)
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Metrics saved to {self.output_dir}")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")


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
        description='Train ResNet18 for Animal Classification (Local & SageMaker)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size (224 or 384)")
    parser.add_argument("--stochastic-depth", type=float, default=0.1,
                        help="Stochastic depth drop rate (for Swin Transformer)")
    
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
    parser.add_argument("--save-file", type=str, default="final_resnet18_model.pth",
                        help="Filename for final saved model")
    
    args = parser.parse_args()
    
    print(f"Arguments: {args}")
    print(f"Initial RAM usage: {get_ram_usage():.2f} MB")
    
    # Detect running environment
    is_sagemaker = os.environ.get("SM_MODEL_DIR") is not None
    
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
        print(f"Initial GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")
    

    
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    
    print(f"Train transform: {train_transform}")
    print(f"Val transform: {val_transform}")
    
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
    print(f"Dataframe: {dataframe.head()}")
    
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
        stratify=np.argmax(dataframe[class_names].values, axis=1), # type: ignore
    )
   
    
    train_dataset = AnimalDataset(train_df, transform=train_transform, folder=train_folder, img_size=img_size)
    val_dataset = AnimalDataset(val_df, transform=val_transform, folder=train_folder, img_size=img_size)

    print(f"After creating datasets RAM usage: {get_ram_usage():.2f} MB")
    if torch.cuda.is_available():
        gpu_alloc, gpu_reserved = get_gpu_memory()
        print(f"After creating datasets GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")

    batch_size = args.batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_cpu
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_cpu
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Load model
    # model = models.swin_t(weights=None)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # print("Initializing Swin Transformer architecture...")
    # model = models.swin_t(weights=None) 
    
    # # 2. Manually adjust Stochastic Depth (The workaround)
    # if args.stochastic_depth != 0.2:
    #     print(f"Adjusting Stochastic Depth to {args.stochastic_depth}...")
    #     for module in model.modules():
    #         if isinstance(module, StochasticDepth):
    #             module.p = args.stochastic_depth

    # 3. Setup the Head
    # model.head = nn.Linear(model.head.in_features, num_classes)
    
    # # 4. LOAD CHECKPOINT
    # checkpoint_dir = os.environ.get("SM_CHANNEL_MODEL")
    
    # print(f"Checking directory: {checkpoint_dir}")
    # if os.path.exists(checkpoint_dir):
    #     files_in_dir = os.listdir(checkpoint_dir) # type: ignore
    #     print("Files found in checkpoint directory:", files_in_dir)
        
    #     # === FIX: AUTO-EXTRACT TAR.GZ ===
    #     if "model.tar.gz" in files_in_dir:
    #         print("Found model.tar.gz! Extracting...")
    #         tar_path = os.path.join(checkpoint_dir, "model.tar.gz")
    #         try:
    #             with tarfile.open(tar_path, "r:gz") as tar:
    #                 tar.extractall(path=checkpoint_dir)
    #             print("✓ Extraction complete.")
    #             print("New directory contents:", os.listdir(checkpoint_dir))
    #         except Exception as e:
    #             print(f"Extraction failed: {e}")
    #     # =================================

    # else:
    #     print("Checkpoint directory does not exist!")

    # # Now look for the .pth file (it should be extracted now)
    # checkpoint_name = "final_swin_t_model_part1_best.pth"
    # checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    
    # further_train = False
    
    # if os.path.exists(checkpoint_path) and further_train:
    #     print(f"LOADING STAGE 1 WEIGHTS FROM: {checkpoint_path}")
    #     state_dict = torch.load(checkpoint_path, map_location=device)
    #     model.load_state_dict(state_dict)
    #     print("✓ Successfully loaded Stage 1 weights.")
    # else:
    #     print(f"⚠️  WARNING: Could not find {checkpoint_name} even after extraction.")
    #     print("Available files:", os.listdir(checkpoint_dir))
    #     print("Falling back to ImageNet-1K weights (Training from Scratch).")
        
    #     # Fallback logic
    #     imagenet_model = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    #     # Transfer weights logic if needed...
    #     model.features = imagenet_model.features
    #     model.norm = imagenet_model.norm
    #     model.permute = imagenet_model.permute

    model = model.to(device)
    
   
    print(f"Model fc: {model.fc}")
    print()
    print(f"After moving model to GPU RAM usage: {get_ram_usage():.2f} MB")
    if torch.cuda.is_available():
        gpu_alloc, gpu_reserved = get_gpu_memory()
        print(f"After moving to GPU - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")
    
    # Mixup configuration (currently disabled)
    mixup_enabled = False
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Optimizer: {optimizer}")
    criterion = nn.CrossEntropyLoss()
    print(f"Criterion: {criterion}")
    print(f"Total Epochs: {args.epochs}")
    
    model.train()
    best_val_acc = 0.0
    print(f"After loading model RAM usage: {get_ram_usage():.2f} MB")

    # scaler = GradScaler()
    
    train_logger = TrainingLogger(output_dir=args.output_dir, class_names=class_names, name="train")
    val_logger = TrainingLogger(output_dir=args.output_dir, class_names=class_names, name="val")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache before each epoch
            gpu_alloc, gpu_reserved = get_gpu_memory()
            print(f"Start of epoch GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")
        
        train_loss, train_acc, train_report = train_epoch(model, train_loader, criterion, optimizer, device, num_classes=num_classes, class_names = class_names)
        train_report['epoch'] = epoch
        train_logger.log_report(train_report)
        train_logger.print_log(train_report)  # Already called inside log_report
        
        if torch.cuda.is_available():
            gpu_alloc, gpu_reserved = get_gpu_memory()
            print(f"After training GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")
        
        val_acc, val_report = validate_epoch(model, val_loader, criterion, device, class_names = class_names)
        val_report['epoch'] = epoch
        val_logger.log_report(val_report)
        val_logger.print_log(val_report)  # Already called inside log_report
        
        if torch.cuda.is_available():
            gpu_alloc, gpu_reserved = get_gpu_memory()
            print(f"After validation GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_file_name = args.save_file.split(".")[0] + "_best.pth"
            print(f"Saving best model to {best_file_name} with val acc {val_acc:.4f}")
            save_path = os.path.join(args.model_dir, best_file_name)
            torch.save(model.state_dict(), save_path)
            print(f"✓ Best model saved! (Val Acc: {val_acc:.4f}), path: {save_path}")


    '''
    Load test data from pickle and run predictions
    '''
    print("\n" + "=" * 60)
    print("RUNNING TEST PREDICTIONS")
    print("=" * 60)
    

        
    train_logger.save()
    val_logger.save()
    print("Saving final model with this name: ", args.save_file)
    save_path = os.path.join(args.model_dir, args.save_file)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")