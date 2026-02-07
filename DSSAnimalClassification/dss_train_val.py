import argparse
import os
import copy
import json
import gc

import pandas as pd
import numpy as np
from PIL import Image
import cv2

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

# from dss_util import get_ram_usage, get_gpu_memory
# from dss_train_val import train_epoch, validate_epoch, TrainingLogger


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler,
    class_names,
    num_classes=8,
    epoch=0,
    total_epochs=1,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []

    num_batches = len(dataloader)
    print(f"\n{'─'*60}")
    print(f"🚂 TRAINING EPOCH {epoch+1}/{total_epochs}")
    print(f"{'─'*60}")
    print(f"   Total batches: {num_batches}")
    print(f"   Batch size: {dataloader.batch_size}")
    print(f"   Total samples: ~{num_batches * dataloader.batch_size}")
    i = 0
    pbar = tqdm(
        dataloader,
        desc=f"Train E{epoch+1}",
        unit="batch",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for batch_idx, (images, labels, ids) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(enabled=(device.type == "cuda")):
            if i == 0:
                print("autocast enabled")
                i += 1
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Guard against NaN poisoning the running average
        batch_loss = loss.item()
        if not np.isnan(batch_loss):
            running_loss += batch_loss * images.size(0)
        else:
            print(
                f"\n   ⚠️ NaN loss detected at batch {batch_idx}! Skipping this batch in avg_loss."
            )
            running_loss += 0.0  # Don't poison running_loss

        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.detach().cpu().numpy().tolist())

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.detach().cpu().numpy().tolist())
        all_ids.extend(ids)
        all_labels.extend(labels.detach().cpu().numpy().tolist())

        # Update progress bar with current metrics
        current_loss = running_loss / total
        current_acc = correct / total
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}",
            }
        )

        # Print detailed info every 25% of batches
        if (batch_idx + 1) % max(1, num_batches // 4) == 0:
            print(
                f"\n   📍 Batch {batch_idx+1}/{num_batches}: Loss={current_loss:.4f}, Acc={current_acc:.4f}"
            )

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    misclassified_list = []

    for i, id in enumerate(all_ids):
        if all_preds[i] != all_labels[i]:
            misclassified_list.append(
                {
                    "id": id,
                    "true_label": class_names[all_labels[i]],
                    "predicted_label": class_names[all_preds[i]],
                    "probability": all_probs[i][all_preds[i]],
                }
            )

    misclassified_images = pd.DataFrame(misclassified_list)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
    report["loss"] = float(running_loss / total)
    report["acc"] = float(correct / total)
    report["ids"] = all_ids
    epoch_acc = float(correct / total)
    report["misclassified_images"] = misclassified_images

    report["log_loss"] = float(log_loss(all_labels, all_probs))

    top3_correct = 0
    for i, label in enumerate(all_labels):
        top3_preds = np.argsort(all_probs[i])[-3:]
        if label in top3_preds:
            top3_correct += 1
    report["top3_accuracy"] = float(top3_correct / len(all_labels))

    class_confidences = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_preds == i
        if class_mask.sum() > 0:
            class_confidences[class_name] = float(all_probs[class_mask, i].mean())
        else:
            class_confidences[class_name] = 0.0
    report["class_confidences"] = class_confidences

    # 4. Ensure macro/weighted metrics are present (sklearn returns np.float64)
    report["macro_precision"] = float(
        precision_score(all_labels, all_preds, average="macro", zero_division=0)
    )
    report["macro_recall"] = float(
        recall_score(all_labels, all_preds, average="macro", zero_division=0)
    )
    report["macro_f1"] = float(
        f1_score(all_labels, all_preds, average="macro", zero_division=0)
    )

    report["micro_precision"] = float(
        precision_score(all_labels, all_preds, average="micro", zero_division=0)
    )
    report["micro_recall"] = float(
        recall_score(all_labels, all_preds, average="micro", zero_division=0)
    )
    report["micro_f1"] = float(
        f1_score(all_labels, all_preds, average="micro", zero_division=0)
    )

    report["weighted_precision"] = float(
        precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    )
    report["weighted_recall"] = float(
        recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    )
    report["weighted_f1"] = float(
        f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    )

    # Verify required keys exist
    assert "macro avg" in report, "macro avg missing from classification_report"
    assert "weighted avg" in report, "weighted avg missing from classification_report"
    assert "accuracy" in report, "accuracy missing from classification_report"
    assert (
        "micro_precision" in report
    ), "micro_precision missing from classification_report"
    assert "micro_recall" in report, "micro_recall missing from classification_report"
    assert "micro_f1" in report, "micro_f1 missing from classification_report"
    assert (
        "weighted_precision" in report
    ), "weighted_precision missing from classification_report"
    assert (
        "weighted_recall" in report
    ), "weighted_recall missing from classification_report"
    assert "weighted_f1" in report, "weighted_f1 missing from classification_report"

    print(f"\n   ✅ Training Complete!")
    print(f"   📊 Final Train Metrics:")
    print(f"      Loss: {report['loss']:.4f}")
    print(f"      Accuracy: {report['acc']:.4f} ({correct}/{total})")
    print(f"      Top-3 Accuracy: {report['top3_accuracy']:.4f}")
    print(f"      Log Loss: {report['log_loss']:.4f}")
    print(f"      Macro F1: {report['macro_f1']:.4f}")
    print(f"      Misclassified: {len(report['misclassified_images'])} samples")

    return epoch_acc, report


def validate_epoch(
    model, dataloader, criterion, device, class_names, epoch=0, total_epochs=1
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []

    num_batches = len(dataloader)
    print(f"\n{'─'*60}")
    print(f"🔍 VALIDATION EPOCH {epoch+1}/{total_epochs}")
    print(f"{'─'*60}")
    print(f"   Total batches: {num_batches}")

    pbar = tqdm(
        dataloader,
        desc=f"Val E{epoch+1}",
        unit="batch",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    with torch.no_grad():
        for batch_idx, (images, labels, ids) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.detach().cpu().numpy().tolist())

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_ids.extend(ids)
            all_preds.extend(predicted.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            current_loss = running_loss / total
            current_acc = correct / total
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.4f}",
                }
            )

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    misclassified_images = pd.DataFrame(
        columns=["id", "true_label", "predicted_label", "probability"]
    )
    for i, id in enumerate(all_ids):
        if all_preds[i] != all_labels[i]:
            misclassified_images.loc[len(misclassified_images)] = {
                "id": id,
                "true_label": class_names[all_labels[i]],
                "predicted_label": class_names[all_preds[i]],
                "probability": all_probs[i][all_preds[i]],
            }

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
    report["loss"] = float(running_loss / total)
    report["acc"] = float(correct / total)
    report["ids"] = all_ids
    epoch_acc = float(correct / total)
    report["misclassified_images"] = misclassified_images

    report["log_loss"] = float(log_loss(all_labels, all_probs))

    top3_correct = 0
    for i, label in enumerate(all_labels):
        top3_preds = np.argsort(all_probs[i])[-3:]
        if label in top3_preds:
            top3_correct += 1
    report["top3_accuracy"] = float(top3_correct / len(all_labels))

    class_confidences = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_preds == i
        if class_mask.sum() > 0:
            class_confidences[class_name] = float(all_probs[class_mask, i].mean())
        else:
            class_confidences[class_name] = 0.0
    report["class_confidences"] = class_confidences

    # 4. Ensure macro/weighted metrics are present (sklearn returns np.float64)
    report["macro_precision"] = float(
        precision_score(all_labels, all_preds, average="macro", zero_division=0)
    )
    report["macro_recall"] = float(
        recall_score(all_labels, all_preds, average="macro", zero_division=0)
    )
    report["macro_f1"] = float(
        f1_score(all_labels, all_preds, average="macro", zero_division=0)
    )

    report["micro_precision"] = float(
        precision_score(all_labels, all_preds, average="micro", zero_division=0)
    )
    report["micro_recall"] = float(
        recall_score(all_labels, all_preds, average="micro", zero_division=0)
    )
    report["micro_f1"] = float(
        f1_score(all_labels, all_preds, average="micro", zero_division=0)
    )

    report["weighted_precision"] = float(
        precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    )
    report["weighted_recall"] = float(
        recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    )
    report["weighted_f1"] = float(
        f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    )

    # Verify required keys exist
    assert "macro avg" in report, "macro avg missing from classification_report"
    assert "weighted avg" in report, "weighted avg missing from classification_report"
    assert "accuracy" in report, "accuracy missing from classification_report"
    assert (
        "micro_precision" in report
    ), "micro_precision missing from classification_report"
    assert "micro_recall" in report, "micro_recall missing from classification_report"
    assert "micro_f1" in report, "micro_f1 missing from classification_report"
    assert (
        "weighted_precision" in report
    ), "weighted_precision missing from classification_report"
    assert (
        "weighted_recall" in report
    ), "weighted_recall missing from classification_report"
    assert "weighted_f1" in report, "weighted_f1 missing from classification_report"

    print(f"\n   ✅ Validation Complete!")
    print(f"   📊 Final Validation Metrics:")
    print(f"      Loss: {report['loss']:.4f}")
    print(f"      Accuracy: {report['acc']:.4f} ({correct}/{total})")
    print(f"      Top-3 Accuracy: {report['top3_accuracy']:.4f}")
    print(f"      Log Loss: {report['log_loss']:.4f}")
    print(f"      Macro F1: {report['macro_f1']:.4f}")
    print(f"      Misclassified: {len(report['misclassified_images'])} samples")

    return epoch_acc, report


class TrainingLogger:
    """Professional logging for SageMaker training"""

    def __init__(
        self,
        data_output_dir,
        name="metrics",
        class_names=None,
        transform_name=None,
    ):
        self.data_output_dir = data_output_dir
        os.makedirs(self.data_output_dir, exist_ok=True)
        self.history = []
        self.class_names = class_names if class_names is not None else []
        self.name = name
        self.transform_name = transform_name

    def log_report(self, report):
        """Log report for one epoch"""
        class_f1s_dict = {
            class_name: report[class_name]["f1-score"]
            for class_name in self.class_names
        }
        class_precisions_dict = {
            class_name: report[class_name]["precision"]
            for class_name in self.class_names
        }
        class_recalls_dict = {
            class_name: report[class_name]["recall"] for class_name in self.class_names
        }
        report["class_f1s"] = class_f1s_dict
        report["class_precisions"] = class_precisions_dict
        report["class_recalls"] = class_recalls_dict
        self.history.append(report)

    def print_log(self, report):
        print(
            f"\n[METRICS] epoch={report['epoch']} {self.name}_loss={report['loss']:.4f} "
            f"{self.name}_acc={report['acc']:.4f}"
        )

        print(f"  Top-3 Accuracy: {report.get('top3_accuracy', 0):.4f}")
        print(f"  Log Loss: {report.get('log_loss', 0):.4f}")

        if "macro avg" in report:
            macro_avg = report["macro avg"]
            print(
                f"  Macro Avg    - F1: {macro_avg['f1-score']:.4f}, "
                f"Precision: {macro_avg['precision']:.4f}, Recall: {macro_avg['recall']:.4f}"
            )

        if "weighted avg" in report:
            weighted_avg = report["weighted avg"]
            print(
                f"  Weighted Avg - F1: {weighted_avg['f1-score']:.4f}, "
                f"Precision: {weighted_avg['precision']:.4f}, Recall: {weighted_avg['recall']:.4f}"
            )

        if "class_f1s" in report and self.class_names:
            print("\n  Per-Class F1 Scores:")
            for class_name in self.class_names:
                f1 = report.get("class_f1s", {}).get(class_name, 0)
                conf = report.get("class_confidences", {}).get(class_name, 0)
                print(f"    {class_name:20s}: F1={f1:.4f}, Confidence={conf:.4f}")

    def save_data_metrics(self, epoch):
        """Save complete history to JSON with proper handling of numpy types"""

        print(f"Saving metrics to {self.data_output_dir}")

        clean_history = []
        for report in self.history:
            clean_report = copy.deepcopy(report)

            if "ids" in clean_report:
                del clean_report["ids"]

            if "misclassified_images" in clean_report:
                if hasattr(clean_report["misclassified_images"], "to_dict"):
                    clean_report["misclassified_images"] = clean_report[
                        "misclassified_images"
                    ].to_dict("records")

            clean_history.append(clean_report)

        json_path = os.path.join(
            self.data_output_dir, f"{self.name}_metrics_{epoch}.json"
        )
        with open(json_path, "w") as f:
            json.dump(clean_history, f, indent=2, cls=NumpyEncoder)

        if clean_history:
            keys = list(clean_history[0].keys())
            df = pd.DataFrame(columns=keys)
            for report in clean_history:
                row = pd.DataFrame([report])
                df = pd.concat([df, row], ignore_index=True)
            df.to_csv(
                os.path.join(self.data_output_dir, f"{self.name}_metrics_{epoch}.csv"),
                index=False,
            )

        print(f"✓ Metrics saved to {self.data_output_dir}")
        print(f"  - {json_path}")
        print(f"  Total epochs logged: {len(clean_history)}")
