# DSS Animal Classification - ML Pipeline Architecture

## Basic resnet18 Model

### Model Configuration

**Model Name**: basic_resnet18_model.pth

**Architecture**: resnet18 with IMAGENET1K_V1 pretrained weights

**Number of Classes**: 8 (antelope_duiker, bird, blank, civet_genet, hog, leopard, monkey_prosimian, rodent)

### Training Hyperparameters

- **Epochs**: 5
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Image Size**: 224x224
- **CUDA**: True
- **Criterion**: nn.CrossEntropyLoss()
- **Optimizer**: optim.AdamW(model.parameters(), lr=args.learning_rate)
- **Learning Rate Scheduler**: None
- **Mixup**: Disabled
- **Train/Val Split**: 75% train / 25% validation

### Data Augmentation

**Train Transforms**:

```python
train_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
```

**Validation Transforms**:

```python
val_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
```

### Model Architecture Details

- **Base Model**: ResNet18 from torchvision.models
- **Pretrained Weights**: IMAGENET1K_V1
- **Final Layer**: Modified FC layer (num_features → 8 classes)
- **Input Shape**: (3, 224, 224)

### Training Details

- **Device**: CUDA (if available)
- **Data Loading**: AnimalDataset (custom PyTorch Dataset)
- **Best Model Selection**: Based on validation accuracy
- **Model Saving**:
  - Final model: basic_resnet18_model.pth
  - Best model: sagemaker_best_resnet18_model.pth

### Results

**Final Epoch Metrics**:
[METRICS] epoch=4 train_loss=0.2881 train_acc=0.90 val_loss=0.4536 val_acc=0.86

**Best Validation Accuracy**: 0.86

### Notes

- This is the baseline model with minimal augmentation
- Uses simple resize transformation without data augmentation
- Training uses ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- No learning rate scheduling applied
- Model saved after all epochs complete

## Basic Resnet50 Model

**Model Name**: basic_resnet50_model.pth

**Architecture**: resnet50 with IMAGENET1K_V1 pretrained weights

**Number of Classes**: 8 (antelope_duiker, bird, blank, civet_genet, hog, leopard, monkey_prosimian, rodent)

### Training Hyperparameters

- **Epochs**: 5
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Image Size**: 224x224
- **CUDA**: True
- **Criterion**: nn.CrossEntropyLoss()
- **Optimizer**: optim.AdamW(model.parameters(), lr=args.learning_rate)
- **Learning Rate Scheduler**: None
- **Mixup**: Disabled
- **Train/Val Split**: 75% train / 25% validation

### Data Augmentation

**Train Transforms**:

```python
train_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
```

**Validation Transforms**:

```python
val_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
```

### Model Architecture Details

- **Base Model**: ResNet50 from torchvision.models
- **Pretrained Weights**: IMAGENET1K_V1
- **Final Layer**: Modified FC layer (num_features → 8 classes)
- **Input Shape**: (3, 224, 224)

### Training Details

- **Device**: CUDA (if available)
- **Data Loading**: AnimalDataset (custom PyTorch Dataset)
- **Best Model Selection**: Based on validation accuracy
- **Model Saving**:
  - Final model: basic_resnet18_model.pth
  - Best model: sagemaker_best_resnet18_model.pth

### Results

**Final Epoch Metrics**:

[METRICS] epoch=4 train_loss=0.2426 train_acc=0.92 val_loss=0.6468 val_acc=0.82

reason for failure overtrained data

**Best Validation Accuracy**: 0.86

### Notes

- This is the baseline model with minimal augmentation
- Uses simple resize transformation without data augmentation
- Training uses ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- No learning rate scheduling applied
- Model saved after all epochs complete

## Swin Transformer (swin_t) Model - Stage 1 Training ✅ COMPLETED

### Model Configuration

**Model Name**: final_swin_t_model_part1_best.pth (best), final_swin_t_model_part1.pth (final)

**Architecture**: Swin Transformer Tiny (swin_t) with IMAGENET1K_V1 pretrained weights

**Number of Classes**: 8 (antelope_duiker, bird, blank, civet_genet, hog, leopard, monkey_prosimian, rodent)

**Model Parameters**: ~28M parameters (3x larger than ResNet18)

### Training Hyperparameters

- **Epochs**: 30 (completed)
- **Batch Size**: 64
- **Learning Rate**: 1e-5 (0.00001)
- **Weight Decay**: 1e-8 (0.00000001)
- **Image Size**: 224x224
- **Stochastic Depth**: 0.2
- **CUDA**: True
- **Criterion**: nn.CrossEntropyLoss()
- **Optimizer**: optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-8)
- **Learning Rate Scheduler**: None
- **Mixup**: Disabled
- **Mixed Precision Training**: Enabled (autocast + GradScaler)
- **Train/Val Split**: 75% train / 25% validation
- **DataLoader Workers**: 4

### Data Augmentation

**Train Transforms**:

```python
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
```

**Validation Transforms**:

```python
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
```

### Model Architecture Details

- **Base Model**: Swin Transformer Tiny from torchvision.models
- **Pretrained Weights**: IMAGENET1K_V1 (ImageNet pretrained)
- **Architecture Type**: Vision Transformer with shifted windows
- **Final Layer**: Modified head (Linear: 768 → 8 classes)
- **Input Shape**: (3, 224, 224)
- **Model Head**: `Linear(in_features=768, out_features=8, bias=True)`
- **Stochastic Depth**: Applied to all transformer blocks (p=0.2)

### Training Details

- **Training Instance**: ml.g4dn.xlarge (Tesla T4, 8 vCPUs, 32 GB RAM)
- **GPU Memory**: 16 GB GDDR6
- **Device**: CUDA (Tesla T4)
- **Data Loading**: AnimalDataset (custom PyTorch Dataset)
- **DataLoader Workers**: 4 (optimized for CPU cores)
- **Best Model Selection**: Based on validation accuracy
- **Model Saving**:
  - Best model: final_swin_t_model_part1_best.pth (saved when val_acc improves)
  - Final model: final_swin_t_model_part1.pth (saved at end of training)
- **Output Path**: s3://sagemaker-us-west-1-253490779227/animal-classification-models_part1
- **Data Source**: S3 bucket (processed data)

### Memory Management & Optimization

- **Mixed Precision Training**: Enabled with `torch.cuda.amp` (autocast + GradScaler)
  - Reduces memory usage by ~40%
  - Speeds up training by ~2-3x on Tensor Cores
- **GPU Memory Tracking**: Enabled at all key checkpoints
- **Memory Cleanup**: `torch.cuda.empty_cache()` before each epoch
- **Garbage Collection**: Enabled (`gc` module)
- **Memory Prints**:
  - Initial state
  - After loading CSV
  - After creating datasets
  - Before/after moving model to GPU
  - Start/end of each epoch

### Results ✅

**Final Epoch Metrics** (Epoch 29/30):

```
[METRICS] epoch=29 train_loss=0.0846 train_acc=0.97 val_loss=0.5178 val_acc=0.89
```

**Best Validation Accuracy**: **89% (0.89)**

**Training Summary**:

- **Training Accuracy**: 97%
- **Validation Accuracy**: 89%
- **Training Loss**: 0.0846
- **Validation Loss**: 0.5178
- **Generalization Gap**: 8% (train_acc - val_acc)

### Performance Analysis

**Strengths**:

- ✅ High validation accuracy (89%) - significant improvement over ResNet models
- ✅ Strong training accuracy (97%) shows model is learning well
- ✅ Swin Transformer architecture effective for fine-grained animal classification
- ✅ Mixed precision training enabled efficient use of GPU memory

**Observations**:

- **Slight Overfitting**: 8% gap between train (97%) and validation (89%) accuracy
- **Validation Loss**: 0.5178 is reasonable but could be improved
- Model shows good generalization despite the gap

**Comparison with Previous Models**:
| Model | Val Accuracy | Train Accuracy | Parameters |
|-------|-------------|----------------|------------|
| ResNet18 (baseline) | 86% | 90% | 11M |
| ResNet50 | 82% | 92% | 25M |
| **Swin-T (Stage 1)** | **89%** ✅ | 97% | 28M |

### Memory Usage Analysis

| Stage                      | RAM Usage | GPU Allocated | GPU Reserved |
| -------------------------- | --------- | ------------- | ------------ |
| Initial                    | ~290 MB   | ~0 MB         | ~0 MB        |
| After CSV Load             | ~290 MB   | ~0 MB         | ~0 MB        |
| After Model Load (CPU)     | ~400 MB   | ~0 MB         | ~0 MB        |
| After Model → GPU          | ~410 MB   | ~600 MB       | ~800 MB      |
| During Training (batch 64) | ~410 MB   | ~4-6 GB       | ~6-8 GB      |

**Note**: Mixed precision training significantly reduced memory usage, allowing batch size of 64 on 16GB GPU.

### Notes

- **Training Strategy**: Single-stage training from ImageNet pretrained weights
- **Why This Works**: Very low learning rate (1e-5) for fine-tuning pretrained features
- **Weight Decay**: Extremely low (1e-8) to preserve pretrained weights
- **Stochastic Depth**: 0.2 provides regularization during training
- **Mixed Precision**: Critical for fitting batch size 64 in 16GB VRAM
- **No Data Augmentation**: Simple resize only - room for improvement
- Uses ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- No learning rate scheduling applied (constant LR throughout)
- Model implements shifted window attention for efficient computation
- Better feature extraction than ResNet for fine-grained classification

### Batch Size & GPU Recommendations

**For ml.g4dn.xlarge (16GB Tesla T4)**:

| Batch Size | Image Size | GPU Memory | Mixed Precision | Status          |
| ---------- | ---------- | ---------- | --------------- | --------------- |
| 64         | 224x224    | ~6-8 GB    | ✅ Enabled      | **Optimal** ✅  |
| 48         | 224x224    | ~5-6 GB    | ✅ Enabled      | Safe            |
| 32         | 224x224    | ~4-5 GB    | ✅ Enabled      | Conservative    |
| 64         | 224x224    | ~12-14 GB  | ❌ Disabled     | **OOM Risk** ❌ |

**Key Insight**: Mixed precision training enables 2x larger batch sizes!

### Training Script Details

- **Filename**: `dss_transformer_train.py`
- **Entry Point**: Used with SageMaker PyTorch Estimator
- **Framework**: PyTorch 2.1 with SageMaker integration
- **Python Version**: py310
- **Data Source**: S3 bucket (processed/train_features)
- **Model Output**: S3 bucket (sagemaker-us-west-1-253490779227)
- **Job Name**: swin-stage1-YYYY-MM-DD-HH-MM-SS-XXX

### Future Improvements

**To Reduce Overfitting (8% gap)**:

1. Add data augmentation (RandomHorizontalFlip, RandomRotation, ColorJitter)
2. Increase stochastic depth to 0.3-0.4
3. Add label smoothing to CrossEntropyLoss
4. Implement learning rate scheduling (CosineAnnealingLR)
5. Try dropout in final layers

**To Boost Performance**:

1. Train longer (50-100 epochs with proper scheduling)
2. Increase image size to 384x384 (adjust batch size to 32)
3. Implement mixup/cutmix augmentation
4. Use larger Swin model (swin_s or swin_b)

## Achieved 74.9% accuracy with base ResNet18. Pipeline in baseArchiecture

## https://arxiv.org/pdf/2201.03545

## Achieved 64.9% accuracy with archiecture and mixup

[METRICS] epoch=4 train_loss=0.9911 train_acc=0.70 val_loss=1.0311 val_acc=0.64

## Achieved 55% accuracy

Changed train transform to RandomResizedCrops because Resize to 224 pixels might remove the small images. Remove GrayScale transformation, as it makes colorJitter inefective. Changed Validation imagae resize to 384 pixels. Removed Schedular as doesn't have an effect in this traing. Schedular only works when I have more than 10 epochs.

Before:

train_transform = transforms.Compose(
[
transforms.Resize((224, 224)),
transforms.Grayscale(num_output_channels=output_channels),
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
transforms.Grayscale(num_output_channels=output_channels),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
)

After:

img_size = 384

train_transform = transforms.Compose(
[
transforms.RandomResizedCrop(
img_size,
scale=(0.75, 1.0),
ratio=(0.9, 1.1)
),
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(10),
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

[METRICS] epoch=4 train_loss=0.8954 train_acc=0.67 val_loss=1.2715 val_acc=0.55

# Achieved 60% accuracy

Changed image size back to 224

[METRICS] epoch=4 train_loss=0.7569 train_acc=0.73 val_loss=1.1257 val_acc=0.60

https://arxiv.org/pdf/2011.11778
