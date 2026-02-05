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
