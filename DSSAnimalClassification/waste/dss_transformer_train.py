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
import tarfile
# import boto3
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
# from sklearn.impute import KNNImputer
# from sklearn.linear_model import LinearRegression
# from sklearn import tree 
# from sklearn.svm import SVC

import torch

import torchvision
from torchvision.models import Swin_T_Weights
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
        """
        Dataset that loads images from numpy arrays stored in DataFrame
        
        Args:
            dataframe: DataFrame with 'image' column (numpy arrays) and label columns
            transform: torchvision transforms to apply
        """
        self.dataframe = dataframe
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
        self.labels = dataframe[label_columns].values.argmax(axis=1)
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

def train_epoch(model, dataloader, criterion, optimizer, device, mixup_enabled=False, num_classes=8, scaler = None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize Scaler for Mixed Precision
    
    
    # pbar = tqdm(dataloader, desc="Training")
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # 1. Cast forward pass to float16 (Auto Mixed Precision)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 2. Scale loss and backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 3. Step optimizer using scaler
        if scaler is not None:
            scaler.step(optimizer)
        else:
            optimizer.step()
        if scaler is not None:
            scaler.update()

        # Statistics (keep existing logic)
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Validation Loss: {epoch_loss:.4f}, Validation Acc: {epoch_acc:.4f}")
    #   pbar.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.4f}")
    return epoch_loss, epoch_acc, all_preds, all_labels

class TrainingLogger:
    """Professional logging for SageMaker training"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = []
    
    def log(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None, lr=None):
        """Log metrics for one epoch"""
        # Store in memory
        metrics = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss) if val_loss else None,
            'val_acc': float(val_acc) if val_acc else None,
            'learning_rate': float(lr) if lr else None
        }
        self.history.append(metrics)
        
        # Print for SageMaker CloudWatch (structured format)
        log_str = f"[METRICS] epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.2f}"
        if val_loss is not None:
            log_str += f" val_loss={val_loss:.4f} val_acc={val_acc:.2f}"
        print(log_str)
    
    def save(self):
        """Save complete history"""
        import json
        import pandas as pd
        
        # Save as JSON (complete data)
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save as CSV (easy viewing)
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.output_dir, 'metrics.csv'), index=False)
        
        print(f"✓ Metrics saved to {self.output_dir}")


if __name__ == "__main__":
    print("Starting training...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)  # Reduced from 32 to 16 for Swin Transformer
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--use-cuda", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--save-file", type=str, default="final_swin_t_model.pth")
    parser.add_argument("--num-cpu", type=int, default=4)
    parser.add_argument( "--image-size", type=int, default=224)
    parser.add_argument("--stochastic-depth", type=float, default=0.1)
    args = parser.parse_args()
    print(f"Arguments: {args}")
    # print(f"⚠️  Note: Swin Transformer requires more memory than ResNet. Batch size set to {args.batch_size}")
    print(f"Initial RAM usage: {get_ram_usage():.2f} MB")
    
    if args.model_dir is None:
        args.model_dir = "/tmp/models"  # Fallback for local testing
        print(f"WARNING: SM_MODEL_DIR not set, using fallback: {args.model_dir}")
    os.makedirs(args.model_dir, exist_ok=True)
    print(f"Model directory: {args.model_dir}")

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
   # model = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    # print("Initializing Swin Transformer architecture...")
    model = models.swin_t(weights=None) 
    
    # 2. Manually adjust Stochastic Depth (The workaround)
    if args.stochastic_depth != 0.2:
        print(f"Adjusting Stochastic Depth to {args.stochastic_depth}...")
        for module in model.modules():
            if isinstance(module, StochasticDepth):
                module.p = args.stochastic_depth

    # 3. Setup the Head
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    # 4. LOAD CHECKPOINT
    checkpoint_dir = os.environ.get("SM_CHANNEL_MODEL")
    
    print(f"Checking directory: {checkpoint_dir}")
    if os.path.exists(checkpoint_dir):
        files_in_dir = os.listdir(checkpoint_dir) # type: ignore
        print("Files found in checkpoint directory:", files_in_dir)
        
        # === FIX: AUTO-EXTRACT TAR.GZ ===
        if "model.tar.gz" in files_in_dir:
            print("Found model.tar.gz! Extracting...")
            tar_path = os.path.join(checkpoint_dir, "model.tar.gz")
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=checkpoint_dir)
                print("✓ Extraction complete.")
                print("New directory contents:", os.listdir(checkpoint_dir))
            except Exception as e:
                print(f"Extraction failed: {e}")
        # =================================

    else:
        print("Checkpoint directory does not exist!")

    # Now look for the .pth file (it should be extracted now)
    checkpoint_name = "final_swin_t_model_part1_best.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    
    further_train = False
    
    if os.path.exists(checkpoint_path) and further_train:
        print(f"LOADING STAGE 1 WEIGHTS FROM: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✓ Successfully loaded Stage 1 weights.")
    else:
        print(f"⚠️  WARNING: Could not find {checkpoint_name} even after extraction.")
        print("Available files:", os.listdir(checkpoint_dir))
        print("Falling back to ImageNet-1K weights (Training from Scratch).")
        
        # Fallback logic
        imagenet_model = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        # Transfer weights logic if needed...
        model.features = imagenet_model.features
        model.norm = imagenet_model.norm
        model.permute = imagenet_model.permute

    model = model.to(device)
    
    print(f"Model: Swin Transformer (swin_t)")
    print(f"Model head: {model.head}")
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
    logger = TrainingLogger()
    model.train()
    best_val_acc = 0.0
    print(f"After loading model RAM usage: {get_ram_usage():.2f} MB")

    scaler = GradScaler()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache before each epoch
            gpu_alloc, gpu_reserved = get_gpu_memory()
            print(f"Start of epoch GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, mixup_enabled=mixup_enabled, num_classes=num_classes, scaler=scaler)
        
        if torch.cuda.is_available():
            gpu_alloc, gpu_reserved = get_gpu_memory()
            print(f"After training GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")
        
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        if torch.cuda.is_available():
            gpu_alloc, gpu_reserved = get_gpu_memory()
            print(f"After validation GPU memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")
        # scheduler.step(val_loss)

        # Save history
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc)

        # Save best model
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
    
    
    logger.save()
    print("Saving final model with this name: ", args.save_file)
    save_path = os.path.join(args.model_dir, args.save_file)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")