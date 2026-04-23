import os
import cv2
import shutil
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

FOLDER_DESCRIPTIONS = {
    "sorted2/geometric": [
        "kyrgyz geometric pattern carpet",
        "shyrdak felt geometric design",
        "central asian geometric textile",
    ],
    "sorted2/animal": [
        "kyrgyz animal motif carpet pattern",
        "ram horn ornament felt textile",
        "zoomorphic kyrgyz pattern rug",
    ],
    "sorted2/symbolic": [
        "kyrgyz symbolic ornament pattern",
        "tunduk sun symbol textile",
        "kyrgyz tribal symbol carpet",
    ],
}

NEGATIVE_DESCRIPTIONS = [
    "person face portrait",
    "landscape nature photo",
    "food drink",
    "building architecture",
    "random unrelated image",
    "screenshot text document",
]

def flatten_folder(folder):
    for root, dirs, files in os.walk(folder):
        if root == folder:
            continue
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                src = os.path.join(root, file)
                dst = os.path.join(folder, file)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
    print(f"  Flattened {folder}")

def remove_duplicates(folder):
    try:
        from imagededup.methods import PHash
        phasher = PHash()
        duplicates = phasher.find_duplicates(
            image_dir=folder,
            num_enc_workers=0,
            num_dist_workers=0
        )
        to_delete = set()
        for original, dups in duplicates.items():
            for dup in dups:
                if dup not in to_delete:
                    to_delete.add(dup)
        for img in to_delete:
            path = os.path.join(folder, img)
            if os.path.exists(path):
                os.remove(path)
        print(f"  Removed {len(to_delete)} duplicates")
    except Exception as e:
        print(f"  Duplicate removal skipped: {e}")

def remove_blurry(folder, threshold=80):
    removed = 0
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        if not os.path.isfile(path):
            continue
        image = cv2.imread(path)
        if image is None:
            os.remove(path)
            removed += 1
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < threshold:
            os.remove(path)
            removed += 1
    print(f"  Removed {removed} blurry images")

def remove_cross_folder_duplicates(folders):
    print("\nRemoving cross-folder duplicates...")
    import hashlib
    seen_hashes = {}
    removed = 0
    for folder in folders:
        if not os.path.exists(folder):
            continue
        for img in os.listdir(folder):
            path = os.path.join(folder, img)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, 'rb') as f:
                    h = hashlib.md5(f.read()).hexdigest()
                if h in seen_hashes:
                    os.remove(path)
                    removed += 1
                else:
                    seen_hashes[h] = path
            except:
                pass
    print(f"  Removed {removed} cross-folder duplicates")

def remove_irrelevant(folder, positive_texts, model, processor, threshold=0.22):
    images = [f for f in os.listdir(folder)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    removed = 0
    kept = 0
    all_texts = positive_texts + NEGATIVE_DESCRIPTIONS

    for i, img in enumerate(images):
        path = os.path.join(folder, img)
        print(f"  Checking {i+1}/{len(images)}...", end="\r")
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(
                text=all_texts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
            positive_score = probs[:len(positive_texts)].max().item()
            if positive_score < threshold:
                os.remove(path)
                removed += 1
            else:
                kept += 1
        except:
            pass

    print(f"\n  Kept: {kept} | Removed: {removed}")

if __name__ == '__main__':
    # Step 1 - Flatten, duplicates, blurry FIRST
    print("Step 1: Flattening, removing duplicates and blurry images...")
    for folder in FOLDER_DESCRIPTIONS.keys():
        if os.path.exists(folder):
            print(f"\nProcessing {folder}...")
            flatten_folder(folder)
            remove_duplicates(folder)
            remove_blurry(folder)
        else:
            print(f"  Folder not found: {folder}")

    # Step 2 - Remove cross folder duplicates
    remove_cross_folder_duplicates(list(FOLDER_DESCRIPTIONS.keys()))

    # Step 3 - Load CLIP once
    print("\nStep 2: Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Model loaded!")

    # Step 4 - Remove irrelevant images
    print("\nStep 3: Removing irrelevant images using CLIP...")
    for folder, descriptions in FOLDER_DESCRIPTIONS.items():
        if os.path.exists(folder):
            print(f"\nProcessing {folder}...")
            remove_irrelevant(folder, descriptions, model, processor)

    # Final count
    print("\n" + "="*40)
    print("Cleaning complete!")
    print("="*40)
    for folder in FOLDER_DESCRIPTIONS.keys():
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder)
                        if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])
            print(f"  {folder}: {count} images")