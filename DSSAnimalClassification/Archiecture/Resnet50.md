
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