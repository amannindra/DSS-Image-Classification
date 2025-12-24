import argparse
import os
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.svm import SVC

import torch

import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

import torchvision.models as models
import os

# import cv2

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os

from PIL import Image
from tqdm import tqdm


class AnimalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
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

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image (numpy array) from DataFrame
        image = self.dataframe.iloc[idx]["image"]
        label = self.labels[idx]

        # Convert numpy array to PIL Image for transforms
        image = Image.fromarray(image.astype('uint8'))

        if self.transform:
            image = self.transform(image)

        return image, label

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
    


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 1

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        # print(images)
        # print(labels)
        images, labels = images.to(device), labels.to(device)

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


def get_image_from_s3(bucket, key, region='us-west-1'):
    """
    Get image from S3 without downloading to disk
    (Optional utility function - not used in main training pipeline)
    """
    try:
        s3_client = boto3.client('s3', region_name=region)
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image = Image.open(BytesIO(response["Body"].read()))
        return image
    except Exception as e:
        print(f"Error loading image from S3: {e}")
        return None

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--use-cuda", type=bool, default=False)

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")
    )
    args = parser.parse_args()
    print(f"Arguments: {args}")
    

    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

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

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    '''
    Loading preprocessed data from pickle file.
    DataFrame contains:
    - filename, width, height, channels, s3_key
    - image: numpy array of pixel data
    - label columns: antelope_duiker, bird, blank, civet_genet, hog, leopard, monkey_prosimian, rodent
    '''
    base_path = args.data_dir
    print(f"Base path: {base_path}")

    # Load from pickle file (contains image arrays)
    train_pkl_path = os.path.join(base_path, "train_data.pkl")
    print(f"Loading training data from {train_pkl_path}...")
    
    # with open(train_pkl_path, 'rb') as f:
    #     dataframe = pickle.load(f)
    
    print(f"DataframeLoaded {len(dataframe)} training samples")
    print(f"Dataframe Columns: {list(dataframe.columns)}")
    print(f"Dataframe sample:\n{dataframe.head()")
    print(f"Dataframe shape: {dataframe.shape}")

    train_df, val_df = train_test_split(
        dataframe,
        test_size=0.25,
        random_state=42,
        stratify=dataframe[class_names].values.argmax(axis=1),
    )
    
    print(f"Train DataFrame sample:\n{train_df.head()}")
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Validation DataFrame sample:\n{val_df.head()}")
    print(f"Validation DataFrame shape: {val_df.shape}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    train_dataset = AnimalDataset(train_df, transform=train_transform)
    val_dataset = AnimalDataset(val_df, transform=val_transform)

    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)
    
    print(f"Model: {model}")
    print(f"Model device: {model.device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        factor=0.5,
    )
    
    
    logger = TrainingLogger()
    model.train()
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Save history
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "sagemaker_best_resnet18_model.pth")
            print(f"✓ Best model saved! (Val Acc: {val_acc:.4f})")
            


    '''
    Load test data from pickle and run predictions
    '''
    print("\n" + "=" * 60)
    print("RUNNING TEST PREDICTIONS")
    print("=" * 60)
    
    test_pkl_path = os.path.join(base_path, "test_data.pkl")
    print(f"Loading test data from {test_pkl_path}...")
    
    # with open(test_pkl_path, 'rb') as f:
    #     test_dataframe = pickle.load(f)
    
    print(f"Loaded {len(test_dataframe)} test samples")
    
    # Run predictions on test set
    model.eval()
    predictions = []
    
    pbar = tqdm(test_dataframe.iterrows(), total=len(test_dataframe), desc="Testing")
    for index, row in pbar:
        print(f"Index: {index}, Row: {row}")
        # Get image from dataframe (numpy array)
        image_array = row["image"]
        image = Image.fromarray(image_array.astype('uint8'))
        
        # Apply validation transform
        image_tensor = val_transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        probs = probabilities.cpu().numpy()[0]  # All class probabilities
        
        # Create prediction record with all class probabilities
        pred_record = {
            'filename': row['filename'],
            'predicted_class': predicted_class,
            'confidence': confidence_score
        }
        
        # Add individual class probabilities (like in dssLocal.ipynb)
        for i, class_name in enumerate(class_names):
            pred_record[f'{class_name}_prob'] = probs[i]
        
        predictions.append(pred_record)
        
        pbar.set_postfix(predicted=predicted_class, conf=f"{confidence_score:.3f}")
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df.head()
    predictions_path = os.path.join(args.model_dir, "test_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    
    print(f"\n✓ Test predictions saved to {predictions_path}")
    print(f"  Columns: {list(predictions_df.columns)}")
    print(f"  Total predictions: {len(predictions_df)}")
    print(f"\nFirst few predictions:")
    print(predictions_df.head())
    
    
    

    
    
    
    # predicted_class, confidence, probs = predict_image( model, test_image_path, val_transform, device, class_names)

     
    logger.save()
    print("Saving model...")
    save_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

