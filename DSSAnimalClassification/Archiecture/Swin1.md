

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