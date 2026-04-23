from pathlib import Path


ROOT = Path(__file__).resolve().parent
LORA_DIR = ROOT / "lora_training"
GEN_DIR = ROOT / "generated_gallery"
API_FILE = ROOT / "api.py"


def verify_lora_training():
    if not LORA_DIR.exists():
        return {"ok": False, "error": "lora_training folder missing"}

    images = [p for p in LORA_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    txts = [p for p in LORA_DIR.iterdir() if p.suffix.lower() == ".txt"]

    if len(images) != 100:
        return {"ok": False, "error": f"expected 100 images, found {len(images)}"}
    if len(txts) != 100:
        return {"ok": False, "error": f"expected 100 captions, found {len(txts)}"}

    for t in txts:
        content = t.read_text(encoding="utf-8").strip().lower()
        if not content.startswith("kyrgyz_ornament"):
            return {"ok": False, "error": f"caption missing trigger token: {t.name}"}

    return {"ok": True}


def verify_generated_gallery():
    if not GEN_DIR.exists():
        return {"ok": False, "error": "generated_gallery folder missing"}
    images = [p for p in GEN_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    counts = {"animal": 0, "geometric": 0, "symbolic": 0}
    for p in images:
        for cls in counts:
            if p.name.startswith(f"{cls}_"):
                counts[cls] += 1
    return {"ok": True, "total": len(images), "counts": counts}


def verify_api_contract():
    if not API_FILE.exists():
        return {"ok": False, "error": "api.py missing"}
    content = API_FILE.read_text(encoding="utf-8")
    required = [
        '"/health"',
        '"/analyze"',
        '"/classify"',
        '"/gallery"',
        '"/generate/gallery"',
        '"/classes"',
        '"/images/{cls}/{filename}"',
    ]
    missing = [r for r in required if r not in content]
    if missing:
        return {"ok": False, "missing_routes": missing}
    return {"ok": True}


def main():
    print("LoRA training folder:", verify_lora_training())
    print("Generated gallery:", verify_generated_gallery())
    print("API contract:", verify_api_contract())


if __name__ == "__main__":
    main()
