import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json, time

# ─── CONFIG ───────────────────────────────────────────────────
TRAIN_DIR  = "dataset_split/train"
VAL_DIR    = "dataset_split/val"
TEST_DIR   = "dataset_split/test"
SAVE_DIR   = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

CLASSES    = ["animal", "geometric", "symbolic"]
IMG_SIZE   = 224
BATCH_SIZE = 8          # safe for 2GB VRAM
EPOCHS_P1  = 15         # phase 1 — head only
EPOCHS_P2  = 10         # phase 2 — fine-tune last blocks
LR_P1      = 1e-3
LR_P2      = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ─── TRANSFORMS ───────────────────────────────────────────────
# Training: heavy augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),        # ← CORRECT: after ToTensor
])

# Val and test: no augmentation — only resize and normalize
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── DATASETS ─────────────────────────────────────────────────
train_dataset = ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset   = ImageFolder(VAL_DIR,   transform=eval_transforms)
test_dataset  = ImageFolder(TEST_DIR,  transform=eval_transforms)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

# ─── WEIGHTED SAMPLER (fixes class imbalance) ─────────────────
class_counts  = np.bincount(train_dataset.targets)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[t] for t in train_dataset.targets]
sampler = WeightedRandomSampler(sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

# ─── MODEL ────────────────────────────────────────────────────
def build_model(num_classes=3):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        drop_rate=0.4,
        drop_path_rate=0.2
    )
    # Replace classifier head
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

model = build_model(num_classes=3).to(DEVICE)

# ─── FREEZE BACKBONE — Phase 1 ────────────────────────────────
def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

def unfreeze_last_blocks(model, num_blocks=2):
    # Unfreeze last N blocks of EfficientNet
    block_names = [f"blocks.{i}" for i in range(6, 8)]
    for name, param in model.named_parameters():
        if any(b in name for b in block_names) or "classifier" in name:
            param.requires_grad = True

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─── LOSS WITH LABEL SMOOTHING ────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ─── TRAIN ONE EPOCH ──────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

# ─── EVALUATE ─────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

# ─── PHASE 1 — Train head only ────────────────────────────────
print("\n" + "="*50)
print("PHASE 1 — Training head only (backbone frozen)")
print("="*50)

freeze_backbone(model)
print(f"Trainable parameters: {count_trainable(model):,}")

optimizer_p1 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_P1, weight_decay=1e-4
)
scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=EPOCHS_P1)

best_val_acc  = 0.0
best_ckpt     = SAVE_DIR / "best_model.pth"
history       = {"phase1": [], "phase2": []}

for epoch in range(1, EPOCHS_P1 + 1):
    t0 = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer_p1)
    val_loss,   val_acc, _, _ = evaluate(model, val_loader)
    scheduler_p1.step()
    elapsed = time.time() - t0

    print(f"[P1 {epoch:02d}/{EPOCHS_P1}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
          f"{elapsed:.1f}s")

    history["phase1"].append({
        "epoch": epoch, "train_loss": train_loss,
        "train_acc": train_acc, "val_acc": val_acc
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_ckpt)
        print(f"  --> Best model saved (val_acc={val_acc:.3f})")

# ─── PHASE 2 — Fine-tune last blocks ──────────────────────────
print("\n" + "="*50)
print("PHASE 2 — Fine-tuning last 2 blocks + head")
print("="*50)

# Load best phase 1 weights before fine-tuning
model.load_state_dict(torch.load(best_ckpt))
unfreeze_last_blocks(model, num_blocks=2)
print(f"Trainable parameters: {count_trainable(model):,}")

optimizer_p2 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_P2, weight_decay=1e-4
)
scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=EPOCHS_P2)

patience, patience_counter = 5, 0

for epoch in range(1, EPOCHS_P2 + 1):
    t0 = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer_p2)
    val_loss,   val_acc, _, _ = evaluate(model, val_loader)
    scheduler_p2.step()
    elapsed = time.time() - t0

    print(f"[P2 {epoch:02d}/{EPOCHS_P2}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
          f"{elapsed:.1f}s")

    history["phase2"].append({
        "epoch": epoch, "train_loss": train_loss,
        "train_acc": train_acc, "val_acc": val_acc
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), best_ckpt)
        print(f"  --> Best model saved (val_acc={val_acc:.3f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

# ─── FINAL TEST EVALUATION ────────────────────────────────────
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

model.load_state_dict(torch.load(best_ckpt))
_, test_acc, preds, labels = evaluate(model, test_loader)

print(f"\nTest Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=CLASSES))
print("Confusion Matrix:")
print(confusion_matrix(labels, preds))

# Save training history
with open("training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nBest model saved to: {best_ckpt}")
print("Training complete.")