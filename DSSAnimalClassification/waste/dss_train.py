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

# import sys
# import pickle
# from io import BytesIO


from sklearn.model_selection import train_test_split

import torch

# import torchvision
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


class AnimalDataset(Dataset):
    def __init__(self, dataframe, transform=None, folder=""):
        """
        Dataset that loads images from numpy arrays stored in DataFrame
        
        Args:
            dataframe: DataFrame with 'image' column (numpy arrays) and label columns
            transform: torchvision transforms to apply
        """
        self.dataframe = dataframe
        self.transform = transform

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
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

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
    
class TestDataset(Dataset):
    def __init__(self, dataframe, transform=None, base_dir=""):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        rel_path = row["filepath"]  # e.g. "test_features/abc.jpg"
        image_path = os.path.join(self.base_dir, rel_path)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, row
    
    # def show_image(self, idx):
    #     image = self.dataframe.iloc[idx]['filepath']


def train_epoch(model, dataloader, criterion, optimizer, device, mixup_enabled=False, num_classes=8):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Mixup is currently disabled - if enabled, mixup_fn would need to be implemented
    # mixup_fn = Mixup(
    #     mixup_alpha=0.2,
    #     cutmix_alpha=0,      # set >0 if you also want CutMix
    #     prob=1.0,              # probability to apply
    #     switch_prob=0.0,
    #     label_smoothing=0.0,   # Changed from 1.0 to 0.0 (1.0 was too high)
    #     num_classes=num_classes
    # )

    # print(f"Mixup enabled: {mixup_enabled}")
    if mixup_enabled:
        raise NotImplementedError("Mixup is not currently implemented. Set mixup_enabled=False or implement Mixup class.")
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        # print(images)
        # print(labels)
        images, labels = images.to(device), labels.to(device)
        # Mixup disabled - if enabled, uncomment and implement mixup_fn above
        # if mixup_enabled:
        #     images, labels = mixup_fn(images, labels)

        # # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        
        # For accuracy calculation with Mixup, use hard labels if available
        if mixup_enabled and labels.dim() > 1:
            # If labels are soft targets (one-hot), convert to hard labels for accuracy
            _, hard_labels = torch.max(labels, 1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == hard_labels).sum().item()
        else:
            # Standard hard labels
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

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


# def get_image_from_s3(bucket, key, region='us-west-1'):
#     """
#     Get image from S3 without downloading to disk
#     (Optional utility function - not used in main training pipeline)
#     """
#     try:
#         s3_client = boto3.client('s3', region_name=region)
#         response = s3_client.get_object(Bucket=bucket, Key=key)
#         image = Image.open(BytesIO(response["Body"].read()))
#         return image
#     except Exception as e:
#         print(f"Error loading image from S3: {e}")
#         return None

def predict_single_image(model, image_array, transform, device, class_names):
    """
    Predict on a single image (numpy array)
    
    Args:
        model: PyTorch model
        image_array: numpy array of image
        transform: torchvision transforms
        device: torch device
        class_names: list of class names
    
    Returns:
        predicted_class, confidence_score, probabilities
    """
    model.eval()

    # Convert numpy to PIL Image
    image = Image.fromarray(image_array.astype('uint8'))
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score, probabilities.cpu().numpy()[0]



if __name__ == "__main__":
    print("Starting training...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--use-cuda", type=lambda x: (str(x).lower() == 'true'), default=True)

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")
    )
    args = parser.parse_args()
    print(f"Arguments: {args}")
    print(f"Initial RAM usage: {get_ram_usage():.2f} MB")
    
    # Validate model directory exists or create it
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
    
    img_size = 224
    
    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Output channels: {output_channels}")
    print(f"Image size: {img_size}")
    
    # Basic resnet18 Model and Resnet50 model
    
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
    
    
    
    
    # Version 1 Base Architecture
    
    # train_transform = transforms.Compose(
    #     [
    #         transforms.Resize((img_size, img_size)),
    #         transforms.Grayscale(num_output_channels=output_channels),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomRotation(10),
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    
    # val_transform = transforms.Compose(
    #     [
    #         transforms.Resize((img_size, img_size)),
    #         transforms.Grayscale(num_output_channels=output_channels),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    
    
    

    # Version 2:
    
    # img_size = 224

    # train_transform = transforms.Compose(
    #     [
    #         transforms.RandomResizedCrop(
    #             img_size,
    #             scale=(0.6, 1.0),
    #             ratio=(0.9, 1.1)
    #         ),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomRotation(10),
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # val_transform = transforms.Compose(
    #     [
    #         transforms.Resize((img_size, img_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    

 
   
    '''
    Loading preprocessed data from pickle file.
    DataFrame contains:
    - filename, width, height, channels, s3_key
    - image: numpy array of pixel data
    - label columns: antelope_duiker, bird, blank, civet_genet, hog, leopard, monkey_prosimian, rodent
    '''
    base_path = args.data_dir
    print(f"Base path: {base_path}")
    
    # Validate base path exists
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Data directory does not exist: {base_path}")

    
    bucket = "animal-classification-dss-works"
    train_folder = os.path.join(base_path, "train_features")
    test_folder = os.path.join(base_path, "test_features")
    train_features_csv = os.path.join(base_path, "train_features.csv")
    test_features_csv = os.path.join(base_path, "test_features.csv")
    train_labels_csv = os.path.join(base_path, "train_labels.csv")
    
    # Validate required files exist
    if not os.path.exists(train_labels_csv):
        raise FileNotFoundError(f"Training labels CSV not found: {train_labels_csv}")
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"Training images folder not found: {train_folder}")
    if not os.path.exists(test_features_csv):
        print(f"Warning: Test features CSV not found: {test_features_csv}")

    dataframe = pd.read_csv(train_labels_csv)
    
    # Validate dataframe is not empty
    if len(dataframe) == 0:
        raise ValueError("Training dataframe is empty!")
    

    print(f"After loading data RAM usage: {get_ram_usage():.2f} MB")
    
    
    
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
    
    print(f"Train DataFrame sample:\n{train_df.head()}") # type: ignore
    print(f"Train DataFrame shape: {train_df.shape}") # type: ignore
    print(f"Validation DataFrame sample:\n{val_df.head()}") # type: ignore
    print(f"Validation DataFrame shape: {val_df.shape}") # type: ignore
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    train_dataset = AnimalDataset(train_df, transform=train_transform, folder=train_folder)
    val_dataset = AnimalDataset(val_df, transform=val_transform, folder=train_folder)

    batch_size = args.batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    
    specific_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    stringform = str(ResNet50_Weights.IMAGENET1K_V2)
    stringform = stringform.replace(".", "")
    print(f"String form: {stringform}")
    model = specific_model
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)
    
    print(f"Model: {model}")
    print()

    # Mixup configuration (currently disabled)
    mixup_enabled = False
    print(f"Mixup enabled: {mixup_enabled}")
    print()
    
    # if mixup_enabled:
    #     criterion = SoftTargetCrossEntropy()
    # else:
    #     criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    print(f"Criterion: {criterion}")
    print()

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"Optimizer: {optimizer}")
    print()
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     patience=2,
    #     factor=0.5,
    # )
    
    logger = TrainingLogger()
    model.train()
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, mixup_enabled=mixup_enabled, num_classes=num_classes)
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        # scheduler.step(val_loss)

        # Save history
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_path = os.path.join(args.model_dir, "sagemaker_best_resnet50model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✓ Best model saved! (Val Acc: {val_acc:.4f}), path: {save_path}")
            


    '''
    Load test data from pickle and run predictions
    '''
    print("\n" + "=" * 60)
    print("RUNNING TEST PREDICTIONS")
    print("=" * 60)
        

        # test_cvs = s3_client.get_object(Bucket=bucket, Key=test_features_csv)
        # test_cvs = test_cvs["Body"].read()
        # csv_buffer = BytesIO(test_cvs)
        # test_dataframe = pd.read_csv(csv_buffer)
        
        # print(test_dataframe.head())
    # with open(test_pkl_path, 'rb') as f:
    #     test_dataframe = pickle.load(f)

    # Load test data if available (optional)
    # if os.path.exists(test_features_csv):
    #     test_dataframe = pd.read_csv(test_features_csv)
    #     print(f"Test dataframe: {test_dataframe.head()}")
    #     print(f"Loaded {len(test_dataframe)} test samples")
    # else:
    #     print(f"Test features CSV not found: {test_features_csv}")
    #     print("Skipping test data loading (this is optional)")
    #     test_dataframe = None
    
    
    logger.save()
    print("Saving model...")
    save_path = os.path.join(args.model_dir, "final_basic_resnet50_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    # if test_dataframe is not None:
    #     print(f"Test samples available: {len(test_dataframe)}")
    
    
    
    # Run predictions on test set
    # test_base = test_folder
    # print(f"Test base: {test_base}")
    # test_dataset = TestDataset(test_dataframe, transform=val_transform, base_dir=test_base)

    # model.eval()
    # predictions = []

    # with torch.no_grad():
    #     for i in tqdm(range(len(test_dataset)), desc="Testing"): # type: ignore
    #         image_tensor, row = test_dataset[i]
    #         image_tensor = image_tensor.unsqueeze(0).to(device)

    #         output = model(image_tensor)
    #         probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    #         pred_idx = int(np.argmax(probs))
    #         confidence = float(probs[pred_idx])

    #         pred_record = {
    #             "filepath": row["filepath"],
    #             "predicted_class": class_names[pred_idx],
    #             "confidence": confidence,
    #         }
    #         for j, cname in enumerate(class_names):
    #             pred_record[f"{cname}_prob"] = float(probs[j])

    #         predictions.append(pred_record)

    # # Save predictions
    # predictions_df = pd.DataFrame(predictions)
    # predictions_df.head()
    # predictions_path = os.path.join(args.model_dir, "test_predictions.csv")
    # predictions_df.to_csv(predictions_path, index=False)
    
    # print(f"\n✓ Test predictions saved to {predictions_path}")
    # print(f"  Columns: {list(predictions_df.columns)}")
    # print(f"  Total predictions: {len(predictions_df)}")
    # print(f"\nFirst few predictions:")
    # print(predictions_df.head())
    
    
    

    
    
    
    # predicted_class, confidence, probs = predict_image( model, test_image_path, val_transform, device, class_names)

     


