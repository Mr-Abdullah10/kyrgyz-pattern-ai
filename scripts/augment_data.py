import cv2
import os
import numpy as np
from pathlib import Path

def augment_image(img):
    augmented = []

    # 1. Horizontal flip
    augmented.append(cv2.flip(img, 1))

    # 2. Vertical flip
    augmented.append(cv2.flip(img, 0))

    # 3. Rotate 90
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

    # 4. Rotate 180
    augmented.append(cv2.rotate(img, cv2.ROTATE_180))

    # 5. Rotate 270
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # 6. Brightness increase
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    augmented.append(bright)

    # 7. Brightness decrease
    dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
    augmented.append(dark)

    # 8. Blur
    augmented.append(cv2.GaussianBlur(img, (5, 5), 0))

    # 9. Zoom in (center crop)
    h, w = img.shape[:2]
    crop = img[h//8:7*h//8, w//8:7*w//8]
    augmented.append(cv2.resize(crop, (w, h)))

    return augmented

def augment_folder(folder, target=300):
    images = [f for f in os.listdir(folder) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
              and not os.path.isdir(os.path.join(folder, f))]
    
    current = len(images)
    print(f"\n{folder}: {current} images found")
    
    if current == 0:
        print(f"  Skipping — no images found")
        return
    
    if current >= target:
        print(f"  Already has {current} images, no augmentation needed")
        return

    count = 0
    i = 0
    while current + count < target:
        img_path = os.path.join(folder, images[i % len(images)])
        img = cv2.imread(img_path)
        if img is None:
            i += 1
            continue
        
        augmented = augment_image(img)
        for j, aug_img in enumerate(augmented):
            if current + count >= target:
                break
            save_path = os.path.join(folder, f"aug_{count}_{j}.jpg")
            cv2.imwrite(save_path, aug_img)
            count += 1
        i += 1

    print(f"  Created {count} new images → total: {current + count}")

if __name__ == '__main__':
    augment_folder("sorted2/geometric", target=300)
    augment_folder("sorted2/animal",    target=300)
    augment_folder("sorted2/symbolic",  target=300)
    print("\nAugmentation complete! Each folder now has 300 images.")