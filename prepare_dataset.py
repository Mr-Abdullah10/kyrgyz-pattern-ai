"""
Dataset Preparation Script for Kyrgyz Pattern AI.

Combines images from multiple sources:
  1. originals/   — 505 manually sorted images
  2. sorted/       — sorted collection (may have augmented copies)
  3. Archive/      — 98 new client images (unsorted, needs auto-classification)

Workflow:
  1. Deduplicate all images by file hash
  2. Auto-classify Archive/ images using the existing EfficientNet model
  3. Merge all unique images
  4. Augment minority classes to balance
  5. Create train/val/test split (80/10/10)

Usage:
  python prepare_dataset.py
"""

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
CLASSES = ["animal", "geometric", "symbolic"]

# ── Output directories ────────────────────────────────────────
MERGED_DIR = ROOT / "merged_unique"
SPLIT_DIR = ROOT / "dataset_split_v2"

# ── Image hashing for deduplication ──────────────────────────
def file_hash(path: Path) -> str:
    """Compute MD5 hash of a file for exact duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path: Path, size=16) -> str:
    """Compute a simple perceptual hash (average hash) for near-duplicate detection."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        img = cv2.resize(img, (size, size))
        avg = img.mean()
        bits = "".join("1" if px > avg else "0" for px in img.flatten())
        return hex(int(bits, 2))
    except:
        return ""


# ── Step 1: Collect and deduplicate from originals + sorted ──
def collect_labeled_images():
    """Gather all labeled images from originals/ and sorted/."""
    labeled = defaultdict(dict)  # class -> {hash: path}
    
    sources = [
        ("originals", ROOT / "originals"),
        ("sorted", ROOT / "sorted"),
    ]
    
    total_scanned = 0
    dupes_skipped = 0
    
    for source_name, source_dir in sources:
        if not source_dir.exists():
            print(f"  [SKIP] {source_name}/ not found")
            continue
            
        for cls in CLASSES:
            cls_dir = source_dir / cls
            if not cls_dir.exists():
                continue
                
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                # Skip augmented images (keep only originals)
                if img_path.name.startswith("aug_"):
                    continue
                    
                total_scanned += 1
                h = file_hash(img_path)
                
                if h not in labeled[cls]:
                    labeled[cls][h] = img_path
                else:
                    dupes_skipped += 1
    
    print(f"  Scanned {total_scanned} labeled images, skipped {dupes_skipped} duplicates")
    for cls in CLASSES:
        print(f"    {cls}: {len(labeled[cls])} unique")
    
    return labeled


# ── Step 2: Auto-classify Archive/ images ────────────────────
def auto_classify_archive(archive_dir: Path) -> dict:
    """Use the existing EfficientNet model to classify unsorted Archive images."""
    print("\n[Step 2] Auto-classifying Archive/ images...")
    
    ckpt_path = ROOT / "checkpoints" / "kyrgyz_classifier_final.pth"
    if not ckpt_path.exists():
        print("  [ERROR] No checkpoint found. Cannot auto-classify.")
        return {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=3)
    model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(model.classifier.in_features, 3))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Deduplicate Archive files first
    archive_files = [
        f for f in archive_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    
    seen_hashes = set()
    unique_files = []
    for f in archive_files:
        h = file_hash(f)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_files.append(f)
    
    print(f"  Archive: {len(archive_files)} files -> {len(unique_files)} unique")
    
    # Classify each image
    classified = defaultdict(list)  # class -> [path]
    confidences = {}
    
    for img_path in unique_files:
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                conf = probs[0, pred_idx].item()
            
            pred_class = CLASSES[pred_idx]
            classified[pred_class].append(img_path)
            confidences[img_path.name] = {
                "predicted": pred_class,
                "confidence": round(conf, 3),
                "all_probs": {
                    CLASSES[i]: round(probs[0, i].item(), 3)
                    for i in range(3)
                }
            }
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
    
    print(f"  Classification results:")
    for cls in CLASSES:
        print(f"    {cls}: {len(classified[cls])} images")
    
    # Save classification report for review
    report_path = ROOT / "archive_classification_report.json"
    with open(report_path, "w") as f:
        json.dump(confidences, f, indent=2)
    print(f"  Report saved: {report_path.name} (review this!)")
    
    return classified


# ── Step 3: Merge all into one clean dataset ─────────────────
def merge_datasets(labeled: dict, archive_classified: dict):
    """Merge labeled data with auto-classified Archive data into merged_unique/."""
    print("\n[Step 3] Merging into merged_unique/...")
    
    # Clean output directory
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    
    stats = {}
    all_hashes = set()
    
    for cls in CLASSES:
        cls_dir = MERGED_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        
        # Copy labeled images
        for h, path in labeled.get(cls, {}).items():
            if h not in all_hashes:
                all_hashes.add(h)
                dst = cls_dir / f"{cls}_{count:04d}{path.suffix}"
                shutil.copy2(path, dst)
                count += 1
        
        # Copy archive images
        for path in archive_classified.get(cls, []):
            h = file_hash(path)
            if h not in all_hashes:
                all_hashes.add(h)
                dst = cls_dir / f"{cls}_{count:04d}{path.suffix}"
                shutil.copy2(path, dst)
                count += 1
        
        stats[cls] = count
        print(f"  {cls}: {count} unique images")
    
    print(f"  Total: {sum(stats.values())} unique images")
    return stats


# ── Step 4: Augment minority classes ─────────────────────────
def augment_to_balance(target_per_class=None):
    """Augment minority classes so all classes have similar counts."""
    print("\n[Step 4] Augmenting to balance classes...")
    
    counts = {}
    for cls in CLASSES:
        cls_dir = MERGED_DIR / cls
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        counts[cls] = len(files)
    
    if target_per_class is None:
        target_per_class = max(counts.values())
    
    print(f"  Target per class: {target_per_class}")
    
    for cls in CLASSES:
        cls_dir = MERGED_DIR / cls
        originals = [f for f in sorted(cls_dir.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and not f.name.startswith("aug_")]
        current = len(list(cls_dir.iterdir()))
        needed = target_per_class - current
        
        if needed <= 0:
            print(f"  {cls}: already has {current} (no augmentation needed)")
            continue
        
        print(f"  {cls}: {current} -> augmenting {needed} more")
        
        aug_count = 0
        idx = 0
        while aug_count < needed:
            src = originals[idx % len(originals)]
            img = cv2.imread(str(src))
            if img is None:
                idx += 1
                continue
            
            augmented = _augment_image(img)
            for j, aug_img in enumerate(augmented):
                if aug_count >= needed:
                    break
                save_path = cls_dir / f"aug_{aug_count:04d}.jpg"
                cv2.imwrite(str(save_path), aug_img)
                aug_count += 1
            idx += 1
        
        print(f"    Created {aug_count} augmented images")


def _augment_image(img):
    """Apply augmentations to a single image."""
    augmented = []
    augmented.append(cv2.flip(img, 1))          # Horizontal flip
    augmented.append(cv2.flip(img, 0))          # Vertical flip
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    augmented.append(cv2.rotate(img, cv2.ROTATE_180))
    
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    augmented.append(bright)
    
    dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
    augmented.append(dark)
    
    augmented.append(cv2.GaussianBlur(img, (5, 5), 0))
    
    h, w = img.shape[:2]
    crop = img[h // 8: 7 * h // 8, w // 8: 7 * w // 8]
    augmented.append(cv2.resize(crop, (w, h)))
    
    return augmented


# ── Step 5: Create train/val/test split ──────────────────────
def create_split(train_ratio=0.80, val_ratio=0.10):
    """Create stratified train/val/test split."""
    print("\n[Step 5] Creating train/val/test split...")
    
    import random
    random.seed(42)
    
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (SPLIT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    split_stats = defaultdict(lambda: defaultdict(int))
    
    for cls in CLASSES:
        cls_dir = MERGED_DIR / cls
        all_images = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        random.shuffle(all_images)
        
        n = len(all_images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_imgs = all_images[:n_train]
        val_imgs = all_images[n_train:n_train + n_val]
        test_imgs = all_images[n_train + n_val:]
        
        for img_path in train_imgs:
            shutil.copy2(img_path, SPLIT_DIR / "train" / cls / img_path.name)
            split_stats["train"][cls] += 1
        
        for img_path in val_imgs:
            shutil.copy2(img_path, SPLIT_DIR / "val" / cls / img_path.name)
            split_stats["val"][cls] += 1
        
        for img_path in test_imgs:
            shutil.copy2(img_path, SPLIT_DIR / "test" / cls / img_path.name)
            split_stats["test"][cls] += 1
    
    print("\n  Split statistics:")
    print(f"  {'':12} {'animal':>8} {'geometric':>10} {'symbolic':>10} {'total':>8}")
    for split in ["train", "val", "test"]:
        total = sum(split_stats[split].values())
        print(f"  {split:12} {split_stats[split]['animal']:8} {split_stats[split]['geometric']:10} {split_stats[split]['symbolic']:10} {total:8}")
    
    total_all = sum(sum(v.values()) for v in split_stats.values())
    print(f"  {'TOTAL':12} {'':8} {'':10} {'':10} {total_all:8}")
    
    return dict(split_stats)


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Kyrgyz Pattern AI — Dataset Preparation")
    print("=" * 60)
    
    print("\n[Step 1] Collecting labeled images from originals/ and sorted/...")
    labeled = collect_labeled_images()
    
    archive_dir = ROOT / "Archive"
    if archive_dir.exists():
        archive_classified = auto_classify_archive(archive_dir)
    else:
        print("\n[Step 2] No Archive/ folder found, skipping.")
        archive_classified = {}
    
    stats = merge_datasets(labeled, archive_classified)
    augment_to_balance()
    split_stats = create_split()
    
    print("\n" + "=" * 60)
    print("DONE! New dataset ready in: dataset_split_v2/")
    print("Review archive_classification_report.json to verify auto-sorted images.")
    print("=" * 60)
