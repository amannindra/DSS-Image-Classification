# Complete SageMaker Pipeline Guide

## What This Does

This pipeline runs **two sequential jobs** on AWS SageMaker:

1. **Preprocessing Job** (PyTorchProcessor) - Extracts image metadata, creates dataframes
2. **Training Job** (PyTorch GPU) - Trains ResNet18 model on GPU

## Files Needed

- `preprocess.py` - Preprocessing script (reads images from S3, creates CSV)
- `train.py` - Training script (trains ResNet18 model)
- `train_with_sagemaker_gpu.py` - **Main script that runs everything**

## How It Works

### Step 1: Preprocessing (PyTorchProcessor)

```python
processor = PyTorchProcessor(
    framework_version='2.1',
    py_version='py310',
    role=ROLE_ARN,
    instance_type='ml.m5.xlarge',  # CPU instance
    instance_count=1
)

processor.run(
    code='preprocess.py',
    inputs=[ProcessingInput(source=S3_INPUT_DATA, ...)],
    outputs=[ProcessingOutput(destination=S3_PREPROCESSED, ...)]
)
```

**What happens:**

1. SageMaker spins up an ml.m5.xlarge instance
2. Loads your `preprocess.py` script
3. Downloads data from `s3://animal-classification-dss-works/data/` to `/opt/ml/processing/input`
4. Runs your script
5. Your script:
   - Reads images from S3
   - Extracts dimensions, metadata
   - Creates train_preprocessed.csv
   - Saves to `/opt/ml/processing/output`
6. SageMaker uploads output to `s3://animal-classification-dss-works/preprocessed/`
7. Shuts down instance

**Cost:** ~$0.50 (30-60 minutes @ $0.23/hour)

### Step 2: Training (PyTorch Estimator)

```python
estimator = PyTorch(
    entry_point='train.py',
    role=ROLE_ARN,
    framework_version='2.1',
    instance_type='ml.g4dn.xlarge',  # GPU instance!
    hyperparameters={'epochs': 10, 'batch-size': 64}
)

estimator.fit({'training': S3_PREPROCESSED})
```

**What happens:**

1. SageMaker spins up ml.g4dn.xlarge (with NVIDIA T4 GPU)
2. Loads your `train.py` script
3. Downloads preprocessed data from S3
4. Runs training with GPU acceleration
5. Saves trained model to S3
6. Shuts down instance

**Cost:** ~$2.00 (1-2 hours @ $0.94/hour)

## How to Run

### Option 1: Run Complete Pipeline

```bash
python train_with_sagemaker_gpu.py
```

This runs **both** preprocessing and training sequentially.

### Option 2: Run Steps Separately

```python
# Just preprocessing
from train_with_sagemaker_gpu import processor
processor.run(...)

# Later, just training
from train_with_sagemaker_gpu import estimator
estimator.fit(...)
```

## File Paths Explained

### In SageMaker Processing:

```
Container File System:
/opt/ml/processing/
├── input/          ← Your S3 data gets downloaded here
│   └── data/
│       ├── train_features/
│       └── test_features/
└── output/         ← Your script saves results here
    ├── train_preprocessed.csv
    └── test_preprocessed.csv
```

**Your script receives:**

- `--input-dir /opt/ml/processing/input`
- `--output-dir /opt/ml/processing/output`

### In SageMaker Training:

```
Container File System:
/opt/ml/
├── input/data/training/    ← Preprocessed data from S3
│   ├── train_preprocessed.csv
│   └── test_preprocessed.csv
├── model/                   ← Save your model here
│   └── model.pth
└── output/                  ← Logs and metrics
```

## Arguments Explained

### preprocess.py Arguments

```python
--input-dir   # Where SageMaker puts your S3 data (default: /opt/ml/processing/input)
--output-dir  # Where you save results (default: /opt/ml/processing/output)
```

**You don't need to change these!** The defaults work with SageMaker's structure.

### ProcessingInput/Output

```python
ProcessingInput(
    source='s3://bucket/data/',              # S3 location
    destination='/opt/ml/processing/input'   # Container location
)

ProcessingOutput(
    source='/opt/ml/processing/output',      # Container location
    destination='s3://bucket/preprocessed/'  # S3 location
)
```

**SageMaker automatically:**

- Downloads S3 data to container
- Uploads container output to S3

## Complete Flow Diagram

```
Local Machine
    │
    │ 1. Run: python train_with_sagemaker_gpu.py
    ├─────────────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│  STEP 1: PREPROCESSING          │    │  STEP 2: TRAINING               │
│  (PyTorchProcessor)             │    │  (PyTorch Estimator)            │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│                                 │    │                                 │
│  Instance: ml.m5.xlarge (CPU)  │    │  Instance: ml.g4dn.xlarge (GPU)│
│  Time: 30-60 min                │    │  Time: 1-2 hours                │
│  Cost: ~$0.50                   │    │  Cost: ~$2.00                   │
│                                 │    │                                 │
│  1. Download from S3            │    │  1. Download preprocessed data  │
│  2. Run preprocess.py           │    │  2. Run train.py                │
│  3. Extract image metadata      │    │  3. Train ResNet18 on GPU       │
│  4. Create CSV files            │    │  4. Save model                  │
│  5. Upload results to S3        │    │  5. Upload to S3                │
│                                 │    │                                 │
└─────────────────────────────────┘    └─────────────────────────────────┘
    │                                             │
    │ Saves to:                                   │ Saves to:
    │ s3://bucket/preprocessed/                   │ s3://bucket/output/model.tar.gz
    ▼                                             ▼
┌────────────────────────────────────────────────────────────┐
│                    Amazon S3                               │
│  ├── data/               (original images)                │
│  ├── preprocessed/       (CSV files from Step 1)          │
│  └── output/             (trained model from Step 2)      │
└────────────────────────────────────────────────────────────┘
```

## Cost Breakdown

| Step          | Instance       | Time      | Cost       |
| ------------- | -------------- | --------- | ---------- |
| Preprocessing | ml.m5.xlarge   | ~45 min   | ~$0.50     |
| Training      | ml.g4dn.xlarge | ~90 min   | ~$2.00     |
| **TOTAL**     |                | ~2.25 hrs | **~$2.50** |

**Compare to:**

- Local CPU: 20-30 hours, FREE but your time!
- EC2 CPU (t3.xlarge): 20+ hours, ~$4/day
- Local with Mac GPU: 6-8 hours, FREE (if you have M1/M2/M3)

## When to Use This

✅ **Use SageMaker Processing + Training when:**

- You have 1000+ images
- Want professional/production pipeline
- Need GPU acceleration
- Don't want to manage infrastructure

❌ **Don't use when:**

- Small dataset (<100 images)
- Just exploring/experimenting
- Happy with local processing time
- Very tight budget (use local/EC2)

## Monitoring Jobs

### Via AWS Console:

1. Go to SageMaker → Processing jobs
2. Go to SageMaker → Training jobs
3. Click on job name to see logs/metrics

### Via Code:

```python
# Logs are streamed to your terminal automatically!
# You'll see:
# - Processing progress
# - Training metrics
# - Any errors
```

## Troubleshooting

### "ResourceLimitExceeded"

You've hit a quota limit. Request increase via AWS Service Quotas.

### "Script not found"

Make sure files are in the same directory as `train_with_sagemaker_gpu.py`.

### "S3 access denied"

Check your IAM role has S3 read/write permissions.

### Processing takes too long

- Use smaller subset for testing
- Check if images are too large
- Consider parallel processing (instance_count > 1)

## Next Steps After Training

1. **Download model:**

   ```python
   model_data = estimator.model_data
   # Download from S3
   ```

2. **Deploy for inference:**

   ```python
   predictor = estimator.deploy(
       instance_type='ml.m5.xlarge',
       initial_instance_count=1
   )
   ```

3. **Batch predictions:**
   ```python
   transformer = estimator.transformer(...)
   transformer.transform(test_data)
   ```

## Summary

**Run this command:**

```bash
python train_with_sagemaker_gpu.py
```

**And SageMaker will:**

1. ✅ Preprocess 16K images (~45 min)
2. ✅ Train ResNet18 on GPU (~90 min)
3. ✅ Save everything to S3
4. ✅ Total cost: ~$2.50

**Much better than:**

- 20-30 hours on local CPU
- Managing EC2 instances manually
- Downloading/uploading data manually

That's it! 🚀
