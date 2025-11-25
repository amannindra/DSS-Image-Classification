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

# ---------------------------------------------------------
# 1. DEFINE YOUR CUSTOM PROCESSING FUNCTIONS
# ---------------------------------------------------------
def pre_process_data(animal_dir, label_idx):
    """
    Reads images and .txt files from the directory to get bounding boxes.
    """
    x_center, y_center, width, height, images = [], [], [], [], []
    
    if not os.path.exists(animal_dir):
        print(f"Warning: Directory {animal_dir} not found.")
        return pd.DataFrame()

    files = os.listdir(animal_dir)
    print(f"Processing {len(files)} files in {animal_dir}")

    for i in files:
        if i.endswith(".txt"):
            # Parse the text file for bounding box
            try:
                with open(os.path.join(animal_dir, i), "r") as file:
                    content = file.read().strip()
                    parts = content.split()
                    # Assuming format: class x_c y_c w h
                    # Your notebook logic had some specific parsing for newlines
                    if len(parts) >= 5:
                        x_center.append(parts[1])
                        y_center.append(parts[2])
                        width.append(parts[3])
                        height.append(parts[4])
                    else:
                        continue
            except Exception as e:
                print(f"Error reading {i}: {e}")
        elif i.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(i)
            
    # Note: This assumes there is 1 .txt for every image and they are processed in order
    # A safer way in production is to match filenames, but we stick to your logic for now.
    # We will truncate to the length of the shortest list to avoid errors
    min_len = min(len(x_center), len(images))
    
    df = pd.DataFrame({
        "x_center": x_center[:min_len],
        "y_center": y_center[:min_len],
        "width": width[:min_len],
        "height": height[:min_len],
        "image": images[:min_len],
    })
    df["animal"] = label_idx
    return df

def resize_and_crop(image_path, xc, yc, w, h):
    """
    Crops the image based on bounding box and resizes to 224x224.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        abs_xc = float(xc) * img_w
        abs_yc = float(yc) * img_h
        abs_w = float(w) * img_w
        abs_h = float(h) * img_h

        x0 = abs_xc - abs_w / 2
        y0 = abs_yc - abs_h / 2
        x1 = abs_xc + abs_w / 2
        y1 = abs_yc + abs_h / 2

        cropped = image.crop((x0, y0, x1, y1))
        resized = cropped.resize((224, 224))
        return np.array(resized)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.zeros((224, 224, 3), dtype=np.uint8)

# ---------------------------------------------------------
# 2. DEFINE THE MODEL (ResNet18 from your notebook)
# ---------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ---------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    
    args = parser.parse_args()
    
    print("Downloading dependencies...")
    # If pandas/PIL are missing in the container, you can pip install here using subprocess
    # but standard PyTorch containers usually have them.

    print(f"Reading data from {args.data_dir}")
    
    # 1. Load Data
    base_path = args.data_dir
    
    # Adjust these folder names if they are different in your S3 bucket
    dfs = []
    classes = {'buffalo': 0, 'elephant': 1, 'rhino': 2, 'zebra': 3}
    
    for class_name, idx in classes.items():
        folder_path = os.path.join(base_path, class_name)
        dfs.append(pre_process_data(folder_path, idx))
        
    data = pd.concat(dfs, ignore_index=True)
    print(f"Total samples found: {len(data)}")

    # 2. Process Images (Resize & Crop)
    processed_images = []
    labels = []
    
    for i in range(len(data)):
        row = data.iloc[i]
        class_name = [k for k, v in classes.items() if v == row['animal']][0]
        img_path = os.path.join(base_path, class_name, row['image'])
        
        img_array = resize_and_crop(
            img_path, 
            row['x_center'], row['y_center'], 
            row['width'], row['height']
        )
        processed_images.append(img_array)
        labels.append(row['animal'])

    X = np.stack(processed_images)
    X = np.transpose(X, (0, 3, 1, 2)) 
    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.tensor(labels).long()
    
    print(f"Data Shape: {X_tensor.shape}")

    # 3. Setup DataLoader
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 4. Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = ResNet18(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(dataloader)}")

    # 5. Save Model
    print("Saving model...")
    save_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    