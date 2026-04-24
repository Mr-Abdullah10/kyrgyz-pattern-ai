"""
Kyrgyz Pattern AI — Multi-Architecture Classifier Training

Trains both ResNet50 and MobileNetV2 classifiers on the Kyrgyz pattern dataset.
Uses transfer learning with a two-phase approach:
  Phase 1: Freeze backbone, train classifier head only
  Phase 2: Unfreeze last layers, fine-tune with lower LR

Usage:
  python Train_models.py                  # Train both models
  python Train_models.py --model resnet50 # Train ResNet50 only
  python Train_models.py --model mobilenet # Train MobileNetV2 only

Each model saves:
  - checkpoints/resnet50_best.pth
  - checkpoints/mobilenet_best.pth
  - checkpoints/resnet50_final.pth  (with full metadata)
  - checkpoints/mobilenet_final.pth (with full metadata)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Handle large images that PIL considers "decompression bombs"
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent
CLASSES = ["animal", "geometric", "symbolic"]

# Use dataset_split_v2 if available, otherwise fall back to dataset_split
DATASET_DIR = ROOT / "dataset_split_v2"
if not DATASET_DIR.exists():
    DATASET_DIR = ROOT / "dataset_split"
    print(f"[INFO] Using fallback dataset: {DATASET_DIR}")

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data Transforms ──────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model Builders ───────────────────────────────────────────
def build_resnet50(num_classes=3):
    """Build ResNet50 with custom classifier head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


def build_mobilenet(num_classes=3):
    """Build MobileNetV2 with custom classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ── Freeze / Unfreeze Helpers ────────────────────────────────
def freeze_backbone(model, arch):
    """Freeze all layers except the classifier head."""
    if arch == "resnet50":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    elif arch == "mobilenet":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def unfreeze_last_layers(model, arch):
    """Unfreeze the last few backbone layers for fine-tuning."""
    if arch == "resnet50":
        # Unfreeze layer4 (last residual block) + fc
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
    elif arch == "mobilenet":
        # Unfreeze last 5 inverted residual blocks + classifier
        features = list(model.features.children())
        for i, block in enumerate(features):
            if i >= len(features) - 5:
                for param in block.parameters():
                    param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ── Training Loop ────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return running_loss / total, correct / total


def evaluate_test(model, loader, device):
    """Full evaluation on test set — returns accuracy, F1, predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASSES)
    
    return acc, f1, cm, report


# ── Main Training Function ───────────────────────────────────
def train_model(arch: str):
    """Train a single model architecture."""
    print("\n" + "=" * 60)
    print(f"Training: {arch.upper()}")
    print(f"Dataset:  {DATASET_DIR}")
    print(f"Device:   {DEVICE}")
    print("=" * 60)
    
    start_time = time.time()
    
    # ── Data loaders ──────────────────────────────────────────
    train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATASET_DIR / "val", transform=val_transform)
    test_dataset = datasets.ImageFolder(DATASET_DIR / "test", transform=val_transform)
    
    # Smaller batch for MX250 (2GB VRAM)
    batch_size = 8 if arch == "resnet50" else 16
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=0, pin_memory=True)
    
    print(f"\n  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print(f"  Batch: {batch_size}")
    
    # ── Build model ───────────────────────────────────────────
    if arch == "resnet50":
        model = build_resnet50(num_classes=3)
    else:
        model = build_mobilenet(num_classes=3)
    model = model.to(DEVICE)
    
    # Class weights for imbalanced data (fast — uses targets list directly)
    from collections import Counter
    label_counts = Counter(train_dataset.targets)
    class_counts = [label_counts.get(i, 1) for i in range(3)]
    weights = torch.FloatTensor([max(class_counts) / c for c in class_counts]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    best_val_acc = 0.0
    best_path = CHECKPOINT_DIR / f"{arch}_best.pth"
    history = {"phase1": [], "phase2": []}
    
    # ── Phase 1: Head only ────────────────────────────────────
    print(f"\n--- Phase 1: Head Only (15 epochs) ---")
    freeze_backbone(model, arch)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=15)
    
    for epoch in range(1, 16):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        history["phase1"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            marker = " * BEST"
        
        print(f"  Epoch {epoch:2d}/15 | "
              f"Train: {train_acc:.3f} ({train_loss:.4f}) | "
              f"Val: {val_acc:.3f} ({val_loss:.4f}){marker}")
    
    # ── Phase 2: Fine-tune ────────────────────────────────────
    print(f"\n--- Phase 2: Fine-tune Last Layers (15 epochs, early stopping) ---")
    
    # Load best from Phase 1
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    unfreeze_last_layers(model, arch)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=15)
    
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, 16):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        history["phase2"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            marker = " * BEST"
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"  Epoch {epoch:2d}/15 | "
              f"Train: {train_acc:.3f} ({train_loss:.4f}) | "
              f"Val: {val_acc:.3f} ({val_loss:.4f}){marker}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    # ── Final Evaluation ──────────────────────────────────────
    print(f"\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    
    test_acc, test_f1, cm, report = evaluate_test(model, test_loader, DEVICE)
    
    elapsed = time.time() - start_time
    
    print(f"\n  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Weighted F1:    {test_f1:.4f}")
    print(f"  Training Time:  {elapsed/60:.1f} minutes")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n  Classification Report:")
    print(report)
    
    # ── Save final checkpoint with metadata ───────────────────
    final_path = CHECKPOINT_DIR / f"{arch}_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": arch,
        "classes": CLASSES,
        "test_accuracy": round(test_acc, 4),
        "weighted_f1": round(test_f1, 4),
        "confusion_matrix": cm.tolist(),
        "input_size": 224,
        "training_time_minutes": round(elapsed / 60, 1),
        "history": history,
        "dataset": str(DATASET_DIR),
    }, final_path)
    
    print(f"\n  Checkpoint saved: {final_path.name}")
    
    return {
        "arch": arch,
        "test_accuracy": round(test_acc, 4),
        "weighted_f1": round(test_f1, 4),
        "training_time": round(elapsed / 60, 1),
        "confusion_matrix": cm.tolist(),
    }


# ── Compare Models ───────────────────────────────────────────
def compare_models(results: list):
    """Print comparison table of all trained models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<15} {'Accuracy':>10} {'F1 Score':>10} {'Time (min)':>12}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
    
    best = max(results, key=lambda r: r["weighted_f1"])
    
    for r in results:
        marker = " <-- WINNER" if r == best else ""
        print(f"  {r['arch']:<15} {r['test_accuracy']:>10.4f} {r['weighted_f1']:>10.4f} {r['training_time']:>12.1f}{marker}")
    
    print(f"\n  Recommended model: {best['arch'].upper()}")
    
    # Save comparison
    comparison_path = ROOT / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump({"models": results, "winner": best["arch"]}, f, indent=2)
    print(f"  Comparison saved: {comparison_path.name}")


# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Kyrgyz Pattern Classifiers")
    parser.add_argument("--model", choices=["resnet50", "mobilenet", "both"],
                        default="both", help="Which model to train")
    args = parser.parse_args()
    
    results = []
    
    if args.model in ("resnet50", "both"):
        results.append(train_model("resnet50"))
    
    if args.model in ("mobilenet", "both"):
        results.append(train_model("mobilenet"))
    
    if len(results) > 1:
        compare_models(results)
    
    print("\nTraining complete!")
