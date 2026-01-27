# ✅ Local Training Setup Complete!

## What Changed

### 1. **Enhanced Argparse in `dss_new_train.py`**
Added comprehensive command-line argument parsing with:
- ✅ Better default values for local training
- ✅ Help text for every argument
- ✅ Automatic directory creation
- ✅ Smart fallbacks (uses environment variables or local defaults)

### 2. **New Files Created**

#### **QUICKSTART_LOCAL.md** ⭐ Start Here!
Quick reference card with:
- Common commands
- Preset configurations
- Hardware-specific recommendations

#### **LOCAL_TRAINING_GUIDE.md** 📖 Full Documentation
Complete guide with:
- All arguments explained
- Example workflows
- Troubleshooting tips
- Performance optimization

#### **run_training.sh** 🚀 Easy Preset Launcher
Shell script with presets:
- `./run_training.sh quick` - Fast test (2 epochs)
- `./run_training.sh full` - Full training (20 epochs)
- `./run_training.sh high-res` - High-res (384x384)
- `./run_training.sh cpu` - CPU training

---

## 🎯 Quick Start - Run Training Now!

### Option 1: Simplest (Default Settings)
```bash
cd DSSAnimalClassification
python dss_new_train.py
```

This will:
- Train for 5 epochs
- Use batch size 16
- Look for data in `./data/`
- Save models to `./models/`
- Save metrics to `./output/`

### Option 2: Custom Settings
```bash
python dss_new_train.py \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --data-dir /path/to/your/data \
    --save-file my_model.pth
```

### Option 3: Using Presets
```bash
chmod +x run_training.sh  # One-time setup
./run_training.sh full    # Run full training
```

---

## 📋 Available Command-Line Arguments

### Training Hyperparameters
```bash
--epochs 20              # Number of training epochs (default: 5)
--batch-size 16          # Batch size (default: 16)
--learning-rate 0.001    # Learning rate (default: 0.001)
--weight-decay 0.01      # L2 regularization (default: 0.01)
--image-size 224         # Image size: 224 or 384 (default: 224)
```

### System Configuration
```bash
--use-cuda true          # Use GPU if available (default: true)
--num-cpu 4              # CPU workers for data loading (default: 4)
```

### Paths
```bash
--data-dir ./data        # Training data location (default: ./data)
--model-dir ./models     # Model save location (default: ./models)
--output-dir ./output    # Metrics save location (default: ./output)
--save-file model.pth    # Model filename (default: final_resnet18_model.pth)
```

### Get Help
```bash
python dss_new_train.py --help
```

---

## 📁 Required Data Structure

Make sure your data directory has this structure:
```
data/
├── train_labels.csv          # Required: CSV with image labels
└── train_features/           # Required: Folder with .jpg images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**train_labels.csv format:**
```csv
id,antelope_duiker,bird,blank,civet_genet,hog,leopard,monkey_prosimian,rodent
image1,0,1,0,0,0,0,0,0
image2,0,0,0,0,0,1,0,0
```

---

## 📊 Output Files

After training, you'll find:

### In `./models/` (or your `--model-dir`):
- **`final_resnet18_model.pth`** - Final model after all epochs
- **`final_resnet18_model_best.pth`** ⭐ - Best model (highest val accuracy)

### In `./output/` (or your `--output-dir`):
- **`train_metrics.csv`** - Training metrics per epoch (open in Excel!)
- **`val_metrics.csv`** - Validation metrics per epoch
- **`train_metrics.json`** - Complete training data (for programmatic analysis)
- **`val_metrics.json`** - Complete validation data

---

## 📈 Metrics Tracked

Every epoch, the following are calculated and saved:

**Basic Metrics:**
- Loss, Accuracy

**Advanced Metrics (NEW!):**
- Log Loss (confidence quality)
- Top-3 Accuracy (correct class in top 3 predictions)
- Per-class F1, Precision, Recall
- Per-class confidence scores
- Confusion matrix

See **METRICS_GUIDE.md** for detailed explanations!

---

## 💻 Example Commands

### Quick Test (2 minutes)
```bash
python dss_new_train.py --epochs 2 --batch-size 8
```

### Full Training
```bash
python dss_new_train.py \
    --epochs 20 \
    --learning-rate 0.0001 \
    --save-file resnet18_v1.pth
```

### High Resolution Training
```bash
python dss_new_train.py \
    --epochs 15 \
    --batch-size 8 \
    --image-size 384 \
    --save-file resnet18_384.pth
```

### CPU Training (no GPU)
```bash
python dss_new_train.py \
    --use-cuda false \
    --batch-size 4 \
    --epochs 10
```

### Custom Data Location
```bash
python dss_new_train.py \
    --data-dir ~/datasets/animals \
    --model-dir ~/models \
    --output-dir ~/logs
```

---

## 🔍 Monitoring Training

### Real-time Console Output
Watch the console for detailed metrics every epoch:
```
Epoch 1/20
Train Loss: 0.3245, Train Acc: 0.8750, Top-3 Acc: 0.9500, Log Loss: 0.4123

[METRICS] epoch=0 train_loss=0.3245 train_acc=0.8750
  Top-3 Accuracy: 0.9500
  Log Loss: 0.4123
  
  Per-Class F1 Scores:
    antelope_duiker     : F1=0.8200, Confidence=0.9100
    bird                : F1=0.7500, Confidence=0.8800
    ...
```

### Analyze Results
```python
import pandas as pd

# Load metrics
df = pd.read_csv('./output/val_metrics.csv')

# View best epoch
best_epoch = df.loc[df['acc'].idxmax()]
print(f"Best validation accuracy: {best_epoch['acc']:.4f} at epoch {best_epoch['epoch']}")
```

---

## 🚀 Next Steps

1. **Start training:**
   ```bash
   python dss_new_train.py --epochs 20
   ```

2. **Monitor progress** - Watch console output

3. **Check results:**
   - Open `./output/val_metrics.csv` in Excel/Numbers
   - Review per-epoch performance

4. **Use best model:**
   - Load `./models/final_resnet18_model_best.pth` for inference
   - This is the model with highest validation accuracy

5. **Analyze metrics:**
   - See **METRICS_GUIDE.md** for understanding metrics
   - Plot training curves from CSV files

---

## 📚 Documentation

- **QUICKSTART_LOCAL.md** - Quick reference (you are here!)
- **LOCAL_TRAINING_GUIDE.md** - Complete guide with examples
- **METRICS_GUIDE.md** - Understanding all metrics
- **S3_STORAGE_GUIDE.md** - For SageMaker training

---

## ⚡ Hardware Recommendations

**If you have a good GPU:**
```bash
python dss_new_train.py --epochs 25 --batch-size 32 --image-size 384
```

**If you have a basic GPU:**
```bash
python dss_new_train.py --epochs 20 --batch-size 16 --image-size 224
```

**If you only have CPU:**
```bash
python dss_new_train.py --epochs 10 --batch-size 4 --use-cuda false
```

---

## ✅ Summary

You can now train locally with:
- ✅ Flexible command-line arguments
- ✅ Comprehensive metrics tracking
- ✅ Easy-to-use presets
- ✅ Automatic directory creation
- ✅ Full documentation

**Just run:** `python dss_new_train.py` and you're training! 🎉
