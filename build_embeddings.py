import torch
import open_clip
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

ORIGINALS_DIR = "originals"
OUTPUT_DIR    = Path("retrieval")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {DEVICE}")

# Load CLIP — best quality model for image retrieval
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(DEVICE).eval()

embeddings = []
records    = []

for cls in ["animal", "geometric", "symbolic"]:
    folder = Path(ORIGINALS_DIR) / cls
    images = list(folder.glob("*.*"))
    print(f"\nEmbedding {cls}: {len(images)} images")

    for img_path in tqdm(images):
        try:
            img    = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize
            embeddings.append(feat.cpu().numpy().squeeze())
            records.append({
                "filename"    : img_path.name,
                "class"       : cls,
                "path"        : str(img_path),
                "pattern_name": "",
                "meaning"     : "",
            })
        except Exception as e:
            print(f"  Skipped {img_path.name}: {e}")

# Save embeddings matrix
emb_matrix = np.array(embeddings).astype("float32")
np.save(OUTPUT_DIR / "embeddings.npy", emb_matrix)

# Save metadata
df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR / "metadata.csv", index=False)

print(f"\nSaved {len(embeddings)} embeddings → retrieval/embeddings.npy")
print(f"Saved metadata → retrieval/metadata.csv")