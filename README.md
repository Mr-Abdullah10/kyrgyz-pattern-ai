# Kyrgyz Pattern AI - Phase 2/3 Quick Run Guide

This guide gives you a copy-paste Colab workflow for LoRA training and gallery generation, then local API startup checks.

## Project Paths

- Local project root: `d:/Projects/kyrgyz/kyrgyz_dataset`
- Colab Drive root used below: `/content/drive/MyDrive/kyrgyz`

## 1) Local Prep (run on your PC)

Generate the LoRA dataset (100 image-caption pairs):

```bash
py -3 prepare_lora_dataset.py
```

Quick validation:

```bash
py -3 verify_phase2_assets.py
```

Expected:
- `LoRA training folder: {'ok': True}`

## 2) Colab One-Go Workflow (copy-paste cells)

Open a fresh notebook in Colab, set runtime to **T4 GPU**.

### Cell A - Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell B - Create working folders

```python
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/kyrgyz")
(ROOT / "lora_training").mkdir(parents=True, exist_ok=True)
(ROOT / "lora_output").mkdir(parents=True, exist_ok=True)
(ROOT / "generated_gallery").mkdir(parents=True, exist_ok=True)
print("Ready:", ROOT)
```

### Cell C - Install packages

```python
!pip install -q diffusers transformers accelerate peft datasets ftfy bitsandbytes
```

Optional: upload `requirements-colab.txt` to Drive and use:

```python
!pip install -q -r /content/drive/MyDrive/kyrgyz/requirements-colab.txt
```

### Cell D - Upload local `lora_training/` folder to Drive

Use the left file pane in Colab and upload your local `lora_training` contents to:

`/content/drive/MyDrive/kyrgyz/lora_training/`

After upload, verify count:

```python
from pathlib import Path

data_dir = Path("/content/drive/MyDrive/kyrgyz/lora_training")
images = [p for p in data_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".webp"]]
captions = [p for p in data_dir.iterdir() if p.suffix.lower() == ".txt"]
print("images:", len(images), "captions:", len(captions))
print("sample caption:", captions[0].read_text(encoding="utf-8")[:120] if captions else "none")
```

Expected:
- `images: 100 captions: 100`
- sample caption starts with `kyrgyz_ornament`

### Cell E - Run LoRA training

```python
import os, subprocess
from pathlib import Path

DATA_DIR = Path("/content/drive/MyDrive/kyrgyz/lora_training")
OUTPUT_DIR = Path("/content/drive/MyDrive/kyrgyz/lora_output")
MODEL_NAME = "runwayml/stable-diffusion-v1-5"

MAX_STEPS = 3000
SAVE_STEPS = 500
RANK = 16
ALPHA = 16
LR = 1e-4
RESOLUTION = 512

train_script = "/usr/local/lib/python3.11/dist-packages/diffusers/examples/text_to_image/train_text_to_image_lora.py"
if not Path(train_script).exists():
    raise FileNotFoundError("Diffusers example script not found. Clone diffusers repo and point to train_text_to_image_lora.py.")

cmd = [
    "accelerate","launch",train_script,
    "--pretrained_model_name_or_path", MODEL_NAME,
    "--train_data_dir", str(DATA_DIR),
    "--resolution", str(RESOLUTION),
    "--center_crop",
    "--random_flip",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--max_train_steps", str(MAX_STEPS),
    "--checkpointing_steps", str(SAVE_STEPS),
    "--learning_rate", str(LR),
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--validation_prompt", "kyrgyz_ornament, geometric pattern, traditional felt textile, high detail",
    "--num_validation_images", "4",
    "--validation_epochs", "1",
    "--rank", str(RANK),
    "--lora_alpha", str(ALPHA),
    "--mixed_precision", "fp16",
    "--output_dir", str(OUTPUT_DIR),
]

print("$", " ".join(cmd))
subprocess.run(cmd, check=True)
print("Training finished:", OUTPUT_DIR)
```

Expected checkpoints during run:
- `checkpoint-500`
- `checkpoint-1000`
- `checkpoint-1500`
- `checkpoint-2000`
- `checkpoint-2500`
- `checkpoint-3000`

Final artifact location should include LoRA weights in:
- `/content/drive/MyDrive/kyrgyz/lora_output`

### Cell F - Resume training if Colab disconnects

```python
import os, subprocess
from pathlib import Path

OUTPUT_DIR = Path("/content/drive/MyDrive/kyrgyz/lora_output")
checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
if not checkpoints:
    raise RuntimeError("No checkpoints found.")
last_ckpt = checkpoints[-1]
print("Resuming from:", last_ckpt)

cmd = [
    "accelerate","launch",
    "/usr/local/lib/python3.11/dist-packages/diffusers/examples/text_to_image/train_text_to_image_lora.py",
    "--pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5",
    "--train_data_dir", "/content/drive/MyDrive/kyrgyz/lora_training",
    "--resolution", "512",
    "--center_crop",
    "--random_flip",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--max_train_steps", "3000",
    "--checkpointing_steps", "500",
    "--learning_rate", "1e-4",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--rank", "16",
    "--lora_alpha", "16",
    "--mixed_precision", "fp16",
    "--output_dir", "/content/drive/MyDrive/kyrgyz/lora_output",
    "--resume_from_checkpoint", str(last_ckpt),
]
print("$", " ".join(cmd))
subprocess.run(cmd, check=True)
```

### Cell G - Generate 50 gallery images

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_DIR = Path("/content/drive/MyDrive/kyrgyz/lora_output")
OUT_DIR = Path("/content/drive/MyDrive/kyrgyz/generated_gallery")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = {
    "geometric": [
        "kyrgyz_ornament, geometric shyrdak pattern, red and black, diamond symmetry, felt textile, high detail",
        "kyrgyz_ornament, geometric carpet motif, zigzag and rhombus repeats, traditional Kyrgyz ornament, high detail",
        "kyrgyz_ornament, mosaic geometric Kyrgyz felt pattern, mirrored layout, handcrafted textile style",
    ],
    "animal": [
        "kyrgyz_ornament, kochkor muyuz horn spirals, zoomorphic ornament, red and black felt textile, high detail",
        "kyrgyz_ornament, deer horn inspired Kyrgyz motif, curved S-shape pattern, traditional carpet design",
        "kyrgyz_ornament, animal-inspired nomadic ornament, swirling horn elements, handcrafted felt style",
    ],
    "symbolic": [
        "kyrgyz_ornament, tunduk sun symbol, radial floral ornament, traditional Kyrgyz felt textile, high detail",
        "kyrgyz_ornament, symbolic yurt crown motif, circular composition, decorative floral geometry",
        "kyrgyz_ornament, sacred Kyrgyz symbolic pattern, sun and flower elements, detailed textile artwork",
    ],
}
COUNTS = {"geometric": 17, "animal": 17, "symbolic": 16}

pipe = DiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, safety_checker=None)
pipe.load_lora_weights(str(LORA_DIR))
pipe = pipe.to("cuda")
gen = torch.Generator(device="cuda").manual_seed(42)

for cls, count in COUNTS.items():
    for i in range(count):
        prompt = PROMPTS[cls][i % len(PROMPTS[cls])]
        image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5, generator=gen).images[0]
        out = OUT_DIR / f"{cls}_{i+1:02d}.jpg"
        image.save(out, quality=95)
        print("saved", out.name)
```

Expected generated counts:
- `geometric`: 17
- `animal`: 17
- `symbolic`: 16
- total: 50

## 3) Bring Gallery Back To Local Project

Copy files from Drive folder:
- `/content/drive/MyDrive/kyrgyz/generated_gallery`

into local folder:
- `d:/Projects/kyrgyz/kyrgyz_dataset/generated_gallery`

## 4) Backend Startup + Smoke Checks

Install API deps if needed:

```bash
py -3 -m pip install -r requirements-api.txt
```

Validate environment before starting backend:

```bash
py -3 check_env.py
```

Run backend:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Test endpoints:
- `GET http://localhost:8000/health`
- `GET http://localhost:8000/classes`
- `GET http://localhost:8000/gallery?cls=geometric&page=1`
- `GET http://localhost:8000/generate/gallery?cls=animal`

## Notes

- Local MX250 is not used for SD training/inference; Colab T4 is required.
- Checkpointing every 500 steps is mandatory for timeout recovery.
- If frontend starts before generation finishes, it can still use `/gallery`; wire `/generate/gallery` once generated images are copied.
