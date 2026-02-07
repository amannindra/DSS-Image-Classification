import argparse
import os
import copy
import json
import gc

import pandas as pd
import numpy as np
from PIL import Image
import cv2


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.ops import StochasticDepth
from torch.cuda.amp import GradScaler, autocast
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.transforms as transforms

import timm
import psutil
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    log_loss,
)

from dss_util import get_ram_usage, get_gpu_memory, get_gpu_memory_nvidia
from dss_datasets import AnimalDatasetConvnext, AnimalDatasetTest
from dss_train_val import train_epoch, validate_epoch, TrainingLogger


def model_compose():
    """Return model configurations (not instantiated models) to allow fresh creation each time"""
    all_models = {
        "model_eva_large_patch14_336.in22k_ft_in22k_in1k": {
            "img_size": 336,
            "batch_size": 16,   
            "timm_name": "eva_large_patch14_336.in22k_ft_in22k_in1k",
            "parameters": 300000000,
        }
    }
    return all_models


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class MultiGPUModel:
    def __init__(self, model, rank, world_size):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        for i in range(world_size):
            print(f"Device ID {i}: {torch.cuda.get_device_name(i)}")
        # DDP takes ONE device per process, not all devices
        self.model = DDP(model, device_ids=[rank])


def create_model(timm_name, num_classes, device):
    """Create a fresh model with pretrained weights"""
    print(f"\n   🔄 Creating fresh model: {timm_name}")
    model = timm.create_model(
        timm_name,
        pretrained=True,
        num_classes=num_classes,
    )

    return model


def transforms_compose(img_size):
    """Create augmentation transforms with the specified image size"""
    all_transforms = {
        "transform_1": A.Compose(
            [
                A.Resize(height=img_size + 20, width=img_size + 20, p=1.0),
                A.RandomCrop(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, p=0.5),
                A.Affine(
                    translate_percent=0.1, scale=(0.8, 1.2), rotate=(-30, 30), p=0.5
                ),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        ),
    }
    return all_transforms


def get_test_transform(img_size):
    """Create test/validation transform using Albumentations"""
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def save_model(model, models_dir, model_name, key, epoch):
    """Save model to models_dir/model_name_key_model_epoch.pth"""
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}_{key}_model_{epoch}.pth")
    torch.save(model.module.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    return model_path


def compute_test_model(
    model,
    models_dir,
    model_name,
    key,
    from_epoch,
    device,
    class_names,
    batch_size,
    num_cpu,
    test_img_size,
    test_folder,
    test_df,
    test_output_dir,
):
    """Run inference on test set and save predictions"""
    model_path = os.path.join(models_dir, f"{model_name}_{key}_model_{from_epoch}.pth")
    # Load into the underlying module if DDP-wrapped, otherwise load directly
    underlying = model.module if hasattr(model, "module") else model
    underlying.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Use Albumentations transform for test (matches training pipeline)
    test_transform = get_test_transform(test_img_size)

    test_dataset = AnimalDatasetTest(
        test_df, transform=test_transform, folder=test_folder, img_size=test_img_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cpu
    )

    rows = []
    os.makedirs(test_output_dir, exist_ok=True)

    pbar = tqdm(test_loader, total=len(test_loader), desc="Testing")
    for batch_idx, (images, ids) in enumerate(pbar):
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

        probs = probs.cpu().numpy()

        for i in range(len(ids)):
            rows.append([ids[i], *probs[i]])

    submission_df = pd.DataFrame(rows, columns=["id"] + class_names)
    output_path = os.path.join(test_output_dir, f"submission_{from_epoch}.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"✓ Test predictions saved to {output_path}")


def main(rank, world_size):
    print("main is called")
    # ── 1. DDP setup FIRST (before any CUDA operations) ──
    ddp_setup(rank, world_size)

    # ── 2. Rank-specific device ──
    device = torch.device(f"cuda:{rank}")
    is_main = (rank == 0)  # Only rank 0 prints/saves

    if is_main:
        print("Starting training...")

    parser = argparse.ArgumentParser(
        description="Train model for Animal Classification (Local & SageMaker)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Input image size (224 or 384)"
    )

    parser.add_argument(
        "--use-cuda",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Use CUDA if available (true/false)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "./data"),
        help="Directory containing training data (train_labels.csv and train_features/)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "./models"),
        help="Directory to save trained models",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
        help="Directory to save metrics and logs",
    )

    # Model saving
    parser.add_argument(
        "--save-file",
        type=str,
        default="final_model.pth",
        help="Filename for final saved model",
    )

    sys_cpus = int(os.environ.get("SM_NUM_CPUS", os.cpu_count()))
    sys_gpus = int(os.environ.get("SM_NUM_GPUS", 1))
    # Divide CPU workers across GPU processes to avoid oversubscription
    cpus_per_gpu = max(1, int(sys_cpus * 2 / 3 / world_size))

    if is_main:
        print(f"System CPUs: {sys_cpus}")
        print(f"System GPUs: {sys_gpus}")
        print(f"CPU workers per GPU process: {cpus_per_gpu}")

    parser.add_argument(
        "--num-cpu",
        type=int,
        default=cpus_per_gpu,
        help="Number of CPU workers for data loading (per GPU process)",
    )

    args = parser.parse_args()

    if is_main:
        print(f"Arguments: {args}")
        print(f"Initial RAM usage: {get_ram_usage():.2f} MB")

    # Detect running environment
    is_sagemaker = os.environ.get("SM_MODEL_DIR") is not None

    if is_main:
        print(f"SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR')}")
        print(f"SM_OUTPUT_DATA_DIR: {os.environ.get('SM_OUTPUT_DATA_DIR')}")

    # Create directories if they don't exist (for local training)
    if not is_sagemaker and is_main:
        os.makedirs(args.model_dir, exist_ok=True)

    if is_main:
        print(f"\n📁 Directory Configuration:")
        print(f"   Environment: {'🚀 SageMaker' if is_sagemaker else '💻 Local'}")
        print(f"   Data directory: {args.data_dir}")
        print(f"   Model directory: {args.model_dir}")

    print(f"[Rank {rank}] Using device: {device}")

    # Validate CUDA availability if requested
    if is_main:
        if args.use_cuda and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
        elif args.use_cuda and torch.cuda.is_available():
            for i in range(world_size):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(
                    f"   GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                )

        print(f"GPU memory: {get_gpu_memory_nvidia()}")

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

    if is_main:
        print(f"Class names: {class_names}")
        print(f"Number of classes: {num_classes}")
        print(f"Initial RAM usage: {get_ram_usage():.2f} MB")

    base_path = args.data_dir
    if is_main:
        print(f"Base path: {base_path}")

    train_folder = os.path.join(base_path, "train_features")
    test_folder = os.path.join(base_path, "test_features")
    test_features_csv = os.path.join(base_path, "test_features.csv")
    train_labels_csv = os.path.join(base_path, "train_labels.csv")

    if not os.path.exists(train_labels_csv):
        raise FileNotFoundError(f"Training labels CSV not found: {train_labels_csv}")
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"Training images folder not found: {train_folder}")
    if not os.path.exists(test_features_csv) and is_main:
        print(f"Warning: Test features CSV not found: {test_features_csv}")

    dataframe = pd.read_csv(train_labels_csv)
    dataframe_test = pd.read_csv(test_features_csv)
    if is_main:
        print(f"Dataframe train: {dataframe.head()}")

    if len(dataframe) == 0:
        raise ValueError("Training dataframe is empty!")

    if is_main:
        print(f"Loaded {len(dataframe)} training samples")
        print(f"Dataframe Columns: {list(dataframe.columns)}")
        print(f"Dataframe shape: {dataframe.shape}")

    train_df, val_df = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=42,
        stratify=np.argmax(dataframe[class_names].values, axis=1),
    )

    # Calculate class weights for handling class imbalance
    train_labels = np.argmax(train_df[class_names].values, axis=1)
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total_samples = len(train_labels)

    # Inverse frequency weighting: weight = total_samples / (num_classes * class_count)
    class_weights = total_samples / (num_classes * class_counts + 1e-6)
    # Move to the rank-specific device
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    if is_main:
        print("class_weights: ", class_weights)
        print(f"\n📊 Class Distribution (Training Set):")
        for i, name in enumerate(class_names):
            print(
                f"   {name:20s}: {class_counts[i]:5d} samples, weight: {class_weights[i].item():.4f}"
            )

    models_dict = model_compose()
    for model_name, model_config in models_dict.items():
        current_img_size = model_config["img_size"]
        timm_name = model_config["timm_name"]
        batch_size = model_config["batch_size"]

        if is_main:
            print(f"\n{'='*60}")
            print(f"🤖 MODEL CONFIG: {model_name}")
            print(f"{'='*60}")
            print(f"   Image size: {current_img_size}x{current_img_size}")
            print(f"   TIMM model: {timm_name}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Create transforms for this model's image size
        augmentation_transform = transforms_compose(current_img_size)
        if is_main:
            print(f"   Transforms: {list(augmentation_transform.keys())}")

        num_transforms = len(augmentation_transform)
        for transform_idx, (transform_key, transform_each) in enumerate(
            augmentation_transform.items()
        ):
            if is_main:
                print(f"\n{'▓'*60}")
                print(f"📦 TRANSFORM {transform_idx+1}/{num_transforms}: {transform_key}")
                print(f"{'▓'*60}")

                print(f"\n🔄 Creating fresh model (pretrained weights)...")

            # ── 3. Create model, move to rank GPU, wrap with DDP ──
            model = create_model(timm_name, num_classes, device)
            model = model.to(device)
            model = DDP(model, device_ids=[rank])

            model.train()
            if is_main:
                print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
                print(
                    f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
                )

            if is_main:
                print(f"\n🔧 Creating datasets...")
            train_dataset = AnimalDatasetConvnext(
                train_df,
                transform=transform_each,
                folder=train_folder,
                img_size=current_img_size,
            )
            val_dataset = AnimalDatasetConvnext(
                val_df,
                transform=get_test_transform(current_img_size),
                folder=train_folder,
                img_size=current_img_size,
            )

            if is_main:
                print(f"   Train dataset size: {len(train_dataset)}")
                print(f"   Val dataset size: {len(val_dataset)}")

            # ── 4. Create samplers (store references for set_epoch) ──
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )

            if is_main:
                print(f"\n🔧 Creating dataloaders...")
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,  # Sampler handles shuffling
                num_workers=args.num_cpu,
                sampler=train_sampler,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_cpu,
                sampler=val_sampler,
                pin_memory=True,
            )

            if is_main:
                print(f"   Train batches: {len(train_loader)}")
                print(f"   Val batches: {len(val_loader)}")
                print(f"   Batch size: {batch_size}")
                print(f"   Num workers: {args.num_cpu}")
                print(f"   RAM usage: {get_ram_usage():.2f} MB")

            # Directory structure: model_dir/model_name/transform_key/{models,data,test}
            base_dir = os.path.join(args.model_dir, model_name, transform_key)
            models_dir = os.path.join(base_dir, "models")
            data_dir = os.path.join(base_dir, "data")
            test_dir = os.path.join(base_dir, "test")

            if is_main:
                os.makedirs(models_dir, exist_ok=True)
                os.makedirs(data_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)

                print(f"\n📁 Output Structure:")
                print(f"   Base: {base_dir}")
                print(f"   Models: {models_dir}")
                print(f"   Data (JSON/CSV): {data_dir}")
                print(f"   Test: {test_dir}")

            train_logger = TrainingLogger(
                data_output_dir=data_dir,
                class_names=class_names,
                name=f"train",
                transform_name=transform_key,
            )
            val_logger = TrainingLogger(
                data_output_dir=data_dir,
                class_names=class_names,
                name=f"val",
                transform_name=transform_key,
            )

            from_epoch = 0
            best_val = 0.0

            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            scaler = GradScaler()
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs)

            if is_main:
                print(f"\n🎯 Starting training loop: {args.epochs} epochs")
                print(f"   Initial LR: {args.learning_rate}")
                print(f"   Scheduler: CosineAnnealingWarmRestarts (T_0={args.epochs})")

            for epoch in range(args.epochs):
                # ── 5. CRITICAL: set_epoch for proper shuffling each epoch ──
                train_sampler.set_epoch(epoch)
                val_sampler.set_epoch(epoch)

                if is_main:
                    print(f"\n{'═'*60}")
                    print(
                        f"🔄 EPOCH {epoch+1}/{args.epochs} | Model: {model_name[:30]}... | Transform: {transform_key}"
                    )
                    print(f"{'═'*60}")

                current_lr = optimizer.param_groups[0]["lr"]
                if is_main:
                    print(f"   Current Learning Rate: {current_lr:.8f}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if is_main:
                        gpu_alloc, gpu_reserved = get_gpu_memory()
                        print(f"GPU memory: {get_gpu_memory_nvidia()}")
                        print(
                            f"   GPU Memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB"
                        )
                if is_main:
                    print(f"   RAM Usage: {get_ram_usage():.2f} MB")

                train_acc, train_report = train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    scaler,
                    class_names,
                    num_classes=num_classes,
                    epoch=epoch,
                    total_epochs=args.epochs,
                )

                train_report["epoch"] = epoch
                train_logger.log_report(train_report)

                if torch.cuda.is_available() and is_main:
                    gpu_alloc, gpu_reserved = get_gpu_memory()
                    print(
                        f"\n   🖥️  Post-train GPU: Allocated={gpu_alloc:.2f} MB, Reserved={gpu_reserved:.2f} MB"
                    )

                val_acc, val_report = validate_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    class_names=class_names,
                    epoch=epoch,
                    total_epochs=args.epochs,
                )
                val_report["epoch"] = epoch

                lr_before = scheduler.get_last_lr()[0]
                scheduler.step()
                lr_after = scheduler.get_last_lr()[0]
                val_report["lr_before"] = lr_before
                val_report["lr_after"] = lr_after

                if is_main:
                    print(f"\n   📈 LR Update: {lr_before:.8f} → {lr_after:.8f}")
                val_logger.log_report(val_report)

                if torch.cuda.is_available() and is_main:
                    gpu_alloc, gpu_reserved = get_gpu_memory()
                    print(f"GPU memory: {get_gpu_memory_nvidia()}")
                    print(
                        f"   🖥️  Post-val GPU: Allocated={gpu_alloc:.2f} MB, Reserved={gpu_reserved:.2f} MB"
                    )

                # Epoch summary
                if is_main:
                    print(f"\n   {'─'*50}")
                    print(f"   📋 EPOCH {epoch+1} SUMMARY:")
                    print(f"      Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                    print(
                        f"      Train Loss: {train_report['loss']:.4f} | Val Loss: {val_report['loss']:.4f}"
                    )
                    print(f"      Best Val Acc so far: {best_val:.4f}")
                    print(f"   {'─'*50}")

                if val_acc > best_val:
                    from_epoch = epoch
                    best_val = val_acc
                    if is_main:
                        print(
                            f"\n   🏆 NEW BEST! Val Accuracy: {val_acc:.4f} (Epoch {epoch+1})"
                        )
                        print(f"   💾 Saving best model...")
                        save_model(model, models_dir, model_name, transform_key, epoch)

            # Training complete for this transform
            if is_main:
                print(f"\n{'▓'*60}")
                print(f"✅ TRAINING COMPLETE for {transform_key}")
                print(f"{'▓'*60}")
                print(f"   Best validation accuracy: {best_val:.4f} (Epoch {from_epoch+1})")

                # Save final metrics and model (rank 0 only)
                print(f"\n💾 Saving final metrics and model...")
                final_epoch_marker = args.epochs + 100
                train_logger.save_data_metrics(final_epoch_marker)
                val_logger.save_data_metrics(final_epoch_marker)
                save_model(model, models_dir, model_name, transform_key, final_epoch_marker)

                # Run test inference (rank 0 only)
                print(f"\n🧪 Running test inference...")
                compute_test_model(
                    model,
                    models_dir,
                    model_name,
                    transform_key,
                    from_epoch,
                    device,
                    class_names,
                    batch_size,
                    args.num_cpu,
                    current_img_size,
                    test_folder,
                    dataframe_test,
                    test_dir,
                )
                print(f"✅ Test inference complete!")

            # Wait for rank 0 to finish saving before cleanup
            torch.distributed.barrier()

            # Clean up model to free GPU memory before next transform
            if is_main:
                print(f"\n🧹 Cleaning up model and freeing memory...")
            del model
            del optimizer
            del scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()
            if is_main:
                print(f"   RAM usage after cleanup: {get_ram_usage():.2f} MB")
                if torch.cuda.is_available():
                    gpu_alloc, gpu_reserved = get_gpu_memory()
                    print(f"GPU memory: {get_gpu_memory_nvidia()}")
                    print(
                        f"   GPU after cleanup: Allocated={gpu_alloc:.2f} MB, Reserved={gpu_reserved:.2f} MB"
                    )

    destroy_process_group()
    if is_main:
        print("\n✓ Training complete!")


if __name__ == "__main__":
    print("STARTING")
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    
