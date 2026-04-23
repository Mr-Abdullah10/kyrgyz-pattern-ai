from pathlib import Path

from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "lora_training"
OUT = ROOT / "generated_gallery"

TARGETS = {"geometric": 17, "animal": 17, "symbolic": 16}


def variant(img: Image.Image, idx: int) -> Image.Image:
    if idx % 3 == 0:
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if idx % 3 == 1:
        return ImageEnhance.Color(img).enhance(1.12)
    return ImageEnhance.Contrast(img).enhance(1.08)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # cleanup old generated images only
    for p in OUT.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            p.unlink()

    for cls, count in TARGETS.items():
        src_imgs = sorted(
            [p for p in SRC.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and p.stem.startswith(f"{cls}_")]
        )
        if not src_imgs:
            raise RuntimeError(f"No source images for {cls} in {SRC}")

        for i in range(count):
            src = src_imgs[i % len(src_imgs)]
            with Image.open(src).convert("RGB") as img:
                out = variant(img, i)
                out.save(OUT / f"{cls}_{i+1:02d}.jpg", quality=95)

    print("Generated bootstrap gallery in:", OUT)


if __name__ == "__main__":
    main()
