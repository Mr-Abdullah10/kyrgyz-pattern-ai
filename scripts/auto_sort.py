import os
import shutil
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

CATEGORIES = {
    "geometric": [
        "kyrgyz geometric pattern with diamonds and triangles",
        "shyrdak felt carpet with sharp angular geometric shapes",
        "central asian textile with zigzag and grid pattern",
        "repeating diamond shapes in traditional carpet",
        "angular geometric pattern with no curves",
    ],
    "animal": [
        "kyrgyz carpet with ram horn spiral scroll pattern",
        "S-shaped curl and scroll motif in felt carpet",
        "zoomorphic animal horn pattern in kyrgyz textile",
        "kochkor muyuz ram horn curved ornament",
        "organic flowing scroll pattern inspired by animal horns",
    ],
    "symbolic": [
        "kyrgyz tunduk circular mandala pattern",
        "circular wheel shape radiating from center like sun",
        "kyrgyz symbolic ornament with central focal point",
        "round medallion pattern with radiating design",
        "sun symbol circular carpet pattern",
    ],
}

ALL_FOLDERS = ["sorted/geometric", "sorted/animal", "sorted/symbolic"]
TEMP_FOLDER = "all_images"

def collect_all_images():
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    total = 0
    for folder in ALL_FOLDERS:
        if not os.path.exists(folder):
            print(f"  Folder not found: {folder}")
            continue
        for img in os.listdir(folder):
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                src = os.path.join(folder, img)
                folder_name = folder.replace("/", "_").replace("\\", "_")
                dst = os.path.join(TEMP_FOLDER, f"{folder_name}_{img}")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                total += 1
    print(f"Collected {total} images into {TEMP_FOLDER}/")
    return total

def classify_image(image_path, model, processor):
    try:
        image = Image.open(image_path).convert("RGB")
        all_texts = []
        text_to_category = {}
        for category, descriptions in CATEGORIES.items():
            for desc in descriptions:
                all_texts.append(desc)
                text_to_category[desc] = category

        inputs = processor(
            text=all_texts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        category_scores = {"geometric": 0, "animal": 0, "symbolic": 0}
        for i, text in enumerate(all_texts):
            cat = text_to_category[text]
            category_scores[cat] += probs[i].item()

        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        return best_category, best_score, category_scores

    except Exception as e:
        return None, 0, {}

def auto_sort():
    # Step 1 - Collect all images
    print("Step 1: Collecting all images from sorted/ folders...")
    total = collect_all_images()
    if total == 0:
        print("No images found!")
        return

    # Step 2 - Load CLIP
    print("\nStep 2: Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Model loaded!")

    # Step 3 - Create output folders
    for folder in ["geometric", "animal", "symbolic"]:
        os.makedirs(f"sorted2/{folder}", exist_ok=True)
    os.makedirs("sorted2/unsorted", exist_ok=True)

    # Step 4 - Classify each image
    print("\nStep 3: Classifying images...")
    images = [f for f in os.listdir(TEMP_FOLDER)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    results = {"geometric": 0, "animal": 0, "symbolic": 0, "skipped": 0}

    for i, img in enumerate(images):
        path = os.path.join(TEMP_FOLDER, img)
        print(f"  [{i+1}/{len(images)}] classifying...", end="\r")

        category, score, all_scores = classify_image(path, model, processor)

        if category and score > 0.25:
            dst = os.path.join("sorted2", category, img)
            shutil.copy2(path, dst)
            results[category] += 1
        else:
            shutil.copy2(path, os.path.join("sorted2/unsorted", img))
            results["skipped"] += 1

    # Step 5 - Clean up temp folder
    shutil.rmtree(TEMP_FOLDER)

    print("\n\nSorting complete!")
    print("="*40)
    print(f"  geometric : {results['geometric']} images")
    print(f"  animal    : {results['animal']} images")
    print(f"  symbolic  : {results['symbolic']} images")
    print(f"  unsorted  : {results['skipped']} images (review manually)")
    print("="*40)
    print("\nFinal sorted images are in the 'sorted2/' folder")

if __name__ == '__main__':
    auto_sort()