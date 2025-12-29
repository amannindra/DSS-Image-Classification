# DSS Animal Classification - ML Pipeline Architecture

## Achieved 74.9% accuracy with base Resnet 

## https://arxiv.org/pdf/2201.03545



## 📋 Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Flow](#data-flow)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Model Architecture](#model-architecture)
7. [S3 Data Structure](#s3-data-structure)
8. [Components Breakdown](#components-breakdown)
9. [Key Features](#key-features)

---

## 🎯 Overview

This is an **end-to-end machine learning pipeline** for classifying wildlife camera trap images into 8 animal categories. The system uses **AWS SageMaker** for distributed training and **PyTorch** with **ResNet18** for deep learning.

### Problem Statement
Classify camera trap images into 8 categories:
- `antelope_duiker`
- `bird`
- `blank`
- `civet_genet`
- `hog`
- `leopard`
- `monkey_prosimian`
- `rodent`

### Technology Stack
- **Cloud Platform**: AWS SageMaker
- **ML Framework**: PyTorch 2.1
- **Model**: ResNet18 (ImageNet pretrained)
- **Storage**: AWS S3
- **Region**: us-west-1

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS S3 Storage                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  s3://animal-classification-dss-works/                   │  │
│  │  ├── data/                                               │  │
│  │  │   ├── train_features/     (raw images)                │  │
│  │  │   ├── test_features/      (raw images)                │  │
│  │  │   ├── train_labels.csv    (one-hot labels)            │  │
│  │  │   └── train_features.csv  (test metadata)             │  │
│  │  └── processed/                                           │  │
│  │      ├── train_features/     (preprocessed images)       │  │
│  │      └── test_features/      (preprocessed images)       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 1: Preprocessing (PyTorchProcessor)          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  preprocess.py                                           │  │
│  │  ├── Check if processed folders exist                    │  │
│  │  ├── Load images from S3                                │  │
│  │  ├── Convert grayscale → RGB                             │  │
│  │  ├── Optimize JPEG compression                            │  │
│  │  └── Upload to processed/ folder                         │  │
│  │                                                           │  │
│  │  Instance: ml.m5.2xlarge (CPU)                           │  │
│  │  Output: Processed images in S3                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 2: Training (PyTorch Estimator)             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  dss_train.py                                            │  │
│  │  ├── Load train_labels.csv                               │  │
│  │  ├── Train/Val split (75/25)                             │  │
│  │  ├── Create AnimalDataset                                │  │
│  │  ├── Initialize ResNet18                                 │  │
│  │  ├── Train for N epochs                                  │  │
│  │  ├── Save best model                                     │  │
│  │  └── Save final model                                    │  │
│  │                                                           │  │
│  │  Instance: ml.m5.large or ml.g4dn.xlarge (GPU)          │  │
│  │  Output: Trained model in S3                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Artifacts (S3)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  s3://sagemaker-us-west-1-{account}/                     │  │
│  │  └── {job-name}/output/model.tar.gz                      │  │
│  │      ├── model.pth                    (final model)      │  │
│  │      ├── sagemaker_best_resnet18_model.pth (best model) │  │
│  │      └── metrics.json                  (training logs)    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### 1. Preprocessing Flow

```
Raw Images (S3)
    │
    ├── data/train_features/*.jpg
    └── data/test_features/*.jpg
    │
    ▼
[preprocess.py]
    │
    ├── Check if processed/ folders exist
    │   └── If exists: Skip processing
    │
    ├── For each image:
    │   ├── Load from S3
    │   ├── Convert grayscale → RGB (if needed)
    │   ├── Optimize JPEG compression
    │   └── Upload to processed/ folder
    │
    ▼
Processed Images (S3)
    │
    ├── processed/train_features/*.jpg
    └── processed/test_features/*.jpg
```

### 2. Training Flow

```
Training Data (S3)
    │
    ├── processed/train_features/  (images)
    └── data/train_labels.csv      (labels)
    │
    ▼
[dss_train.py]
    │
    ├── Load train_labels.csv
    │   └── Columns: id, antelope_duiker, bird, blank, ...
    │
    ├── Train/Val Split (75/25, stratified)
    │
    ├── Create AnimalDataset
    │   ├── Load images from local filesystem
    │   ├── Apply transforms (resize, normalize)
    │   └── Convert one-hot labels → class indices
    │
    ├── Initialize ResNet18
    │   ├── Pretrained: ImageNet weights
    │   └── Final layer: 8 classes
    │
    ├── Training Loop (N epochs)
    │   ├── Forward pass
    │   ├── Compute loss (CrossEntropyLoss)
    │   ├── Backward pass
    │   ├── Optimizer step (AdamW)
    │   ├── Validation
    │   └── Save best model (if val_acc improves)
    │
    ▼
Trained Model (S3)
    │
    ├── model.pth
    └── sagemaker_best_resnet18_model.pth
```

---

## 🔧 Preprocessing Pipeline

### File: `preprocess.py`

**Purpose**: Prepare raw images for training by standardizing format and optimizing storage.

### Key Functions

#### `get_image_from_s3(bucket, key)`
- **Input**: S3 bucket name, object key
- **Output**: PIL Image object
- **Function**: Downloads image from S3 without saving to disk (memory-efficient)

#### `get_all_image_keys(bucket, prefix)`
- **Input**: S3 bucket, folder prefix
- **Output**: List of image keys
- **Function**: Lists all `.jpg`, `.jpeg`, `.png` files in S3 folder

#### `process_images(bucket, image_keys, is_test)`
- **Input**: Bucket, list of image keys, test flag
- **Output**: None (uploads to S3)
- **Process**:
  1. Load image from S3
  2. Convert grayscale → RGB (if needed)
  3. Optimize JPEG compression
  4. Upload to `processed/train_features/` or `processed/test_features/`

#### `folder_exists_and_not_empty(bucket, path)`
- **Input**: S3 bucket, folder path
- **Output**: Boolean
- **Function**: Checks if processed folder exists and contains files (avoids reprocessing)

### Processing Logic

```python
# Pseudocode
if processed folders exist:
    skip processing
else:
    for each image in train_features/:
        load from S3
        convert grayscale → RGB if needed
        optimize JPEG
        upload to processed/train_features/
    
    for each image in test_features/:
        load from S3
        convert grayscale → RGB if needed
        optimize JPEG
        upload to processed/test_features/
```

### Memory Management
- Processes images one at a time (not batch loading)
- Uses `BytesIO` for in-memory image manipulation
- Monitors RAM usage with `get_ram_usage()`

---

## 🎓 Training Pipeline

### File: `dss_train.py`

**Purpose**: Train ResNet18 model on preprocessed images with labels.

### Key Components

#### 1. **AnimalDataset Class**
```python
class AnimalDataset(Dataset):
    - Loads images from local filesystem
    - Converts one-hot labels → class indices
    - Applies transforms (resize, normalize, augment)
```

**Key Methods**:
- `__getitem__(idx)`: Returns (image_tensor, label_index)
- Loads image: `id + ".jpg"` from `train_features/` folder
- Applies transforms: Resize(224x224), Grayscale→RGB, Normalize

#### 2. **Data Transforms**

**Training Transforms**:
```python
- Resize(224, 224)
- Grayscale(num_output_channels=3)  # Convert to RGB
- RandomHorizontalFlip()              # Data augmentation
- RandomRotation(10)                 # Data augmentation
- ColorJitter(brightness=0.2, ...)   # Data augmentation
- ToTensor()
- Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Validation Transforms**:
```python
- Resize(224, 224)
- Grayscale(num_output_channels=3)
- ToTensor()
- Normalize(...)  # Same as training
```

#### 3. **Model Architecture**

```python
ResNet18 (Pretrained)
    │
    ├── Backbone: ResNet18 (ImageNet weights)
    │   └── Feature extraction layers
    │
    └── Classifier: Linear(512 → 8)
        └── Final layer for 8 animal classes
```

#### 4. **Training Configuration**

- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `AdamW` (learning_rate=0.001)
- **Scheduler**: `ReduceLROnPlateau`
  - Mode: `min` (monitor validation loss)
  - Patience: 2 epochs
  - Factor: 0.5 (reduce LR by 50%)
- **Batch Size**: 32 (configurable)
- **Epochs**: 5 (configurable)
- **Train/Val Split**: 75/25 (stratified)

#### 5. **Training Loop**

```python
for epoch in range(epochs):
    # Training phase
    train_loss, train_acc = train_epoch(...)
    
    # Validation phase
    val_loss, val_acc = validate_epoch(...)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Logging
    logger.log(epoch, train_loss, train_acc, val_loss, val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        save_model("sagemaker_best_resnet18_model.pth")
```

#### 6. **TrainingLogger Class**

- Logs metrics to JSON and CSV
- Saves to `/opt/ml/output/data/` (SageMaker output directory)
- Files: `metrics.json`, `metrics.csv`

---

## 🧠 Model Architecture

### ResNet18 Architecture

```
Input: (3, 224, 224) RGB Image
    │
    ▼
Conv2d(3 → 64, kernel=7, stride=2)
    │
    ▼
BatchNorm + ReLU
    │
    ▼
MaxPool2d
    │
    ▼
Residual Block 1 (2 layers, 64 channels)
    │
    ▼
Residual Block 2 (2 layers, 128 channels)
    │
    ▼
Residual Block 3 (2 layers, 256 channels)
    │
    ▼
Residual Block 4 (2 layers, 512 channels)
    │
    ▼
AdaptiveAvgPool2d(1, 1)  → (512,)
    │
    ▼
Linear(512 → 8)  → Output logits
    │
    ▼
Softmax → Class probabilities
```

### Model Details

- **Base Model**: ResNet18 (torchvision)
- **Pretrained Weights**: ImageNet1K_V1
- **Input Size**: 224×224×3
- **Output Size**: 8 (one per class)
- **Parameters**: ~11.7M total
- **Trainable Parameters**: ~4.1M (final layer + fine-tuning)

---

## 📦 S3 Data Structure

```
s3://animal-classification-dss-works/
│
├── data/
│   ├── train_features/          # Raw training images
│   │   ├── ZJ000001.jpg
│   │   ├── ZJ000002.jpg
│   │   └── ...
│   │
│   ├── test_features/           # Raw test images
│   │   ├── ZJ000501.jpg
│   │   └── ...
│   │
│   ├── train_labels.csv         # Training labels (one-hot)
│   │   Columns: id, antelope_duiker, bird, blank, ...
│   │
│   └── train_features.csv      # Test metadata
│       Columns: id, filepath
│
└── processed/
    ├── train_features/          # Preprocessed training images
    │   ├── ZJ000001.jpg
    │   └── ...
    │
    └── test_features/           # Preprocessed test images
        └── ...
```

### SageMaker Model Output

```
s3://sagemaker-us-west-1-{account-id}/
└── {training-job-name}/
    └── output/
        └── model.tar.gz
            ├── model.pth                        # Final model weights
            ├── sagemaker_best_resnet18_model.pth # Best model weights
            └── code/                            # Inference code (if any)
```

---

## 🧩 Components Breakdown

### Preprocessing Components

| Component | File | Purpose |
|-----------|------|---------|
| S3 Image Loader | `get_image_from_s3()` | Load images without disk I/O |
| Image Processor | `process_images()` | Convert format, optimize compression |
| Folder Checker | `folder_exists_and_not_empty()` | Avoid redundant processing |
| RAM Monitor | `get_ram_usage()` | Track memory usage |

### Training Components

| Component | File | Purpose |
|-----------|------|---------|
| Dataset | `AnimalDataset` | Load images + labels, apply transforms |
| Model | `resnet18()` | ResNet18 with ImageNet weights |
| Trainer | `train_epoch()` | Training loop with progress bar |
| Validator | `validate_epoch()` | Validation loop with metrics |
| Logger | `TrainingLogger` | Save metrics to JSON/CSV |
| Model Saver | `torch.save()` | Save best + final models |

### SageMaker Integration

| Component | Purpose |
|-----------|---------|
| `PyTorchProcessor` | Run preprocessing on managed instance |
| `PyTorch` Estimator | Run training on managed instance |
| `SM_MODEL_DIR` | SageMaker model output directory |
| `SM_CHANNEL_TRAINING` | SageMaker training data input |
| `SM_OUTPUT_DATA_DIR` | SageMaker metrics output |

---

## ✨ Key Features

### 1. **Idempotent Preprocessing**
- Checks if processed folders exist before processing
- Avoids redundant work and costs

### 2. **Memory-Efficient Image Loading**
- Uses `BytesIO` for in-memory processing
- Processes images one at a time
- Monitors RAM usage

### 3. **Stratified Train/Val Split**
- Maintains class distribution in train/val sets
- Prevents data leakage

### 4. **Data Augmentation**
- Random horizontal flips
- Random rotations (±10°)
- Color jitter (brightness, contrast, saturation)
- Only applied during training (not validation)

### 5. **Model Checkpointing**
- Saves best model based on validation accuracy
- Saves final model after all epochs
- Both models uploaded to S3 automatically

### 6. **Comprehensive Logging**
- Real-time progress bars (tqdm)
- Epoch-level metrics (loss, accuracy)
- Saves metrics to JSON and CSV
- CloudWatch logs for SageMaker jobs

### 7. **Grayscale Image Handling**
- Converts grayscale images to RGB (3 channels)
- Ensures compatibility with pretrained ResNet18

### 8. **GPU Support**
- Automatically detects CUDA availability
- Falls back to CPU if GPU not available
- Configurable via `--use-cuda` flag

---

## 🔄 Execution Flow

### Step 1: Preprocessing Job
```python
processor = PyTorchProcessor(...)
processor.run(code='preprocess.py')
```
**Result**: Processed images in `s3://bucket/processed/`

### Step 2: Training Job
```python
estimator = PyTorch(
    entry_point='dss_train.py',
    source_dir='.',
    ...
)
estimator.fit({'training': s3_processed_data})
```
**Result**: Trained model in `s3://sagemaker-bucket/{job-name}/output/`

---

## 📊 Performance Considerations

### Preprocessing
- **Instance**: `ml.m5.2xlarge` (CPU)
- **Cost**: ~$0.23/hour
- **Time**: ~30-60 minutes (depends on image count)

### Training
- **Instance**: `ml.m5.large` (CPU) or `ml.g4dn.xlarge` (GPU)
- **Cost**: 
  - CPU: ~$0.13/hour
  - GPU: ~$0.74/hour
- **Time**: 
  - CPU: ~2-4 hours (5 epochs)
  - GPU: ~30-60 minutes (5 epochs)

### Memory Usage
- Preprocessing: Monitored with `get_ram_usage()`
- Training: Batch size 32, 2 DataLoader workers

---

## 🚀 Usage

### Run Preprocessing
```python
python preprocess.py
# Or via SageMaker:
processor.run(code='preprocess.py')
```

### Run Training
```python
python dss_train.py \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --data-dir /opt/ml/input/data/training \
    --model-dir /opt/ml/model
# Or via SageMaker:
estimator.fit({'training': s3_data_path})
```

---

## 📝 Notes

- **Image Format**: All images converted to RGB JPEG
- **Image Size**: Resized to 224×224 during training
- **Labels**: One-hot encoded in CSV, converted to indices in Dataset
- **Model**: ResNet18 pretrained on ImageNet, fine-tuned for 8 classes
- **Output**: Model weights saved as `.pth` files

---

## 🔗 Related Files

- `preprocess.py` - Preprocessing script
- `dss_train.py` - Training script
- `SageAnimalTrain.ipynb` - SageMaker orchestration notebook
- `manage_sagemaker_jobs.py` - Job management utility

---

**Last Updated**: 2025-01-XX
**Version**: 1.0

