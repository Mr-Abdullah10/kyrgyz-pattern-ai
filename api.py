import base64
import io
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

import faiss
import numpy as np
import open_clip
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv_models
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
DATASET_SPLIT = ROOT / "dataset_split"
RETRIEVAL_DIR = ROOT / "retrieval"
GENERATED_DIR = ROOT / "generated_gallery"

CLASSES = ["animal", "geometric", "symbolic"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Available model checkpoints
MODEL_REGISTRY = {
    "resnet50": {
        "path": ROOT / "checkpoints" / "resnet50_final.pth",
        "display": "ResNet50",
    },
    "mobilenet": {
        "path": ROOT / "checkpoints" / "mobilenet_final.pth",
        "display": "MobileNetV2",
    },
    "efficientnet": {
        "path": ROOT / "checkpoints" / "kyrgyz_classifier_final.pth",
        "display": "EfficientNet-B0",
    },
}

clf_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

app = FastAPI(title="Kyrgyz Pattern AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state: dict[str, Any] = {}


def build_model_by_arch(arch_key: str) -> nn.Module:
    """Build a model architecture without loading weights."""
    if arch_key == "resnet50":
        model = tv_models.resnet50(weights=None)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(in_feat, 512),
            nn.ReLU(inplace=True), nn.Dropout(p=0.3),
            nn.Linear(512, 3),
        )
    elif arch_key == "mobilenet":
        model = tv_models.mobilenet_v2(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(in_feat, 256),
            nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(256, 3),
        )
    else:  # efficientnet
        model = timm.create_model(
            "efficientnet_b0", pretrained=False,
            drop_rate=0.4, drop_path_rate=0.2,
        )
        in_feat = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_feat, 3))
    return model


def load_classifier(arch_key: str) -> torch.nn.Module:
    """Load a classifier by architecture key."""
    info = MODEL_REGISTRY[arch_key]
    model = build_model_by_arch(arch_key)

    ckpt = torch.load(info["path"], map_location=DEVICE, weights_only=False)
    model_state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(model_state)
    model.to(DEVICE).eval()
    return model


def load_clip():
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return clip_model.to(DEVICE).eval(), preprocess


def classify_image(pil_img: Image.Image, model_key: str = None):
    """Classify using the specified model (or default)."""
    if model_key and model_key in state.get("models", {}):
        model = state["models"][model_key]
    else:
        # Use first available model as default
        model = next(iter(state.get("models", {}).values()))
    tensor = clf_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    top_idx = int(np.argmax(probs))
    return CLASSES[top_idx], float(probs[top_idx]), probs.tolist()


def get_clip_embedding(pil_img: Image.Image):
    tensor = state["clip_preprocess"](pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = state["clip_model"].encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


def retrieve_similar(embedding: np.ndarray, top_k: int = 3):
    scores, indices = state["faiss_index"].search(embedding, top_k)
    metadata = state["metadata"]
    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = metadata.iloc[int(idx)]
        pattern_name = row.get("pattern_name", "")
        meaning = row.get("meaning", "")
        if pd.isna(pattern_name):
            pattern_name = ""
        if pd.isna(meaning):
            meaning = ""
        results.append(
            {
                "filename": row.get("filename", ""),
                "class": row.get("class", ""),
                "path": row.get("path", ""),
                "pattern_name": str(pattern_name),
                "meaning": str(meaning),
                "similarity": float(score),
            }
        )
    return results


def open_uploaded_image(upload: UploadFile) -> Image.Image:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


@app.on_event("startup")
def startup_event():
    # Load all available classifier models
    state["models"] = {}
    state["model_meta"] = {}
    for key, info in MODEL_REGISTRY.items():
        if info["path"].exists():
            try:
                state["models"][key] = load_classifier(key)
                ckpt = torch.load(info["path"], map_location="cpu", weights_only=False)
                state["model_meta"][key] = {
                    "display_name": info["display"],
                    "accuracy": ckpt.get("test_accuracy"),
                    "f1": ckpt.get("weighted_f1"),
                }
                print(f"  Loaded: {info['display']}")
            except Exception as e:
                print(f"  Failed to load {info['display']}: {e}")

    state["default_model"] = next(iter(state["models"]), None)
    state["clip_model"], state["clip_preprocess"] = load_clip()
    state["faiss_index"] = faiss.read_index(str(RETRIEVAL_DIR / "faiss.index"))
    state["metadata"] = pd.read_csv(RETRIEVAL_DIR / "metadata.csv")
    state["embeddings"] = np.load(RETRIEVAL_DIR / "embeddings.npy").astype("float32")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classes")
def classes():
    metadata = state["metadata"]
    counts = {
        cls: int((metadata["class"] == cls).sum())
        for cls in CLASSES
    }
    return {"classes": CLASSES, "counts": counts}


@app.get("/models")
def list_models():
    """List all available classifier models."""
    return {
        "available": state.get("model_meta", {}),
        "default": state.get("default_model"),
    }


@app.post("/classify")
def classify(
    file: UploadFile = File(...),
    model: str = Query(default=None, description="Model key: resnet50, mobilenet, or efficientnet"),
):
    pil_img = open_uploaded_image(file)
    model_key = model or state.get("default_model")
    pred_class, confidence, probabilities = classify_image(pil_img, model_key)
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": dict(zip(CLASSES, probabilities)),
        "model_used": MODEL_REGISTRY.get(model_key, {}).get("display", model_key),
    }


@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    model: str = Query(default=None, description="Model key: resnet50, mobilenet, or efficientnet"),
):
    pil_img = open_uploaded_image(file)
    model_key = model or state.get("default_model")
    pred_class, confidence, probabilities = classify_image(pil_img, model_key)
    emb = get_clip_embedding(pil_img)
    similar = retrieve_similar(emb, top_k=3)
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": dict(zip(CLASSES, probabilities)),
        "model_used": MODEL_REGISTRY.get(model_key, {}).get("display", model_key),
        "low_confidence_warning": confidence < 0.60,
        "similar_patterns": similar,
    }


@app.get("/gallery")
def gallery(
    cls: str = Query("all"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
):
    metadata = state["metadata"].copy()
    if cls != "all":
        if cls not in CLASSES:
            raise HTTPException(status_code=400, detail=f"Invalid cls '{cls}'")
        metadata = metadata[metadata["class"] == cls]

    total = len(metadata)
    start = (page - 1) * page_size
    end = start + page_size
    rows = metadata.iloc[start:end]
    items = rows.to_dict(orient="records")

    return {
        "class_filter": cls,
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items,
    }


@app.get("/generate/gallery")
def generated_gallery(cls: str = Query("all")):
    if not GENERATED_DIR.exists():
        return {"class_filter": cls, "items": []}

    allowed = CLASSES if cls == "all" else [cls]
    if cls != "all" and cls not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid cls '{cls}'")

    files = []
    for c in allowed:
        for path in sorted(GENERATED_DIR.glob(f"{c}_*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            files.append(
                {
                    "class": c,
                    "filename": path.name,
                    "url": f"/generated-images/{path.name}",
                }
            )
    return {"class_filter": cls, "items": files}


@app.get("/images/{cls}/{filename}")
def dataset_image(cls: str, filename: str):
    if cls not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid class '{cls}'")
    root = DATASET_SPLIT / "train" / cls
    path = root / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/generated-images/{filename}")
def generated_image(filename: str):
    path = GENERATED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Generated image not found")
    return FileResponse(path)


# ── REAL-TIME GENERATION ──────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate a new Kyrgyz pattern image from a text prompt in real time."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    try:
        from generate_service import generate_image

        image, filepath, backend, enhanced_prompt = generate_image(req.prompt)

        # Encode image as base64 for JSON response
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image_base64": img_b64,
            "filename": Path(filepath).name,
            "url": f"/generated-images/{Path(filepath).name}",
            "enhanced_prompt": enhanced_prompt,
            "backend": backend,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
