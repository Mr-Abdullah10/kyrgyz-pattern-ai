import csv
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "labels.csv"
OUTPUT_DIR = ROOT / "lora_training"

TARGET_COUNTS = {
    "geometric": 34,
    "animal": 33,
    "symbolic": 33,
}

CLASS_PROMPTS = {
    "geometric": "geometric pattern, red and black, diamond grid, mirror symmetry, traditional felt textile, high detail",
    "animal": "animal motif pattern, red and black, kochkor muyuz horn spirals, curved ornament, traditional felt textile, high detail",
    "symbolic": "symbolic floral pattern, red and black, tunduk sun motif, radial ornament, traditional felt textile, high detail",
}


def image_score(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            w, h = img.size
            return (w * h, min(w, h))
    except Exception:
        return (0, 0)


def collect_candidates() -> dict[str, list[dict]]:
    candidates = defaultdict(list)
    with LABELS.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row["class"].strip().lower()
            split = row["split"].strip().lower()
            filename = row["filename"].strip()
            rel_path = row["path"].replace("\\", "/").strip()

            if cls not in TARGET_COUNTS:
                continue
            if split != "train":
                continue
            if filename.startswith("aug_"):
                continue

            src_path = ROOT / rel_path
            if not src_path.exists():
                continue

            area, min_side = image_score(src_path)
            if area == 0:
                continue

            candidates[cls].append(
                {
                    "filename": filename,
                    "src_path": src_path,
                    "area": area,
                    "min_side": min_side,
                }
            )
    return candidates


def caption_for(cls: str) -> str:
    return f"kyrgyz_ornament, {CLASS_PROMPTS[cls]}"


def main() -> None:
    candidates = collect_candidates()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cls, target in TARGET_COUNTS.items():
        ranked = sorted(
            candidates[cls],
            key=lambda x: (x["area"], x["min_side"], x["filename"]),
            reverse=True,
        )
        selected = ranked[:target]
        if len(selected) < target:
            raise RuntimeError(f"Not enough images for class '{cls}': {len(selected)} < {target}")

        for idx, item in enumerate(selected, start=1):
            src = item["src_path"]
            ext = src.suffix.lower()
            out_stem = f"{cls}_{idx:03d}"
            dst_img = OUTPUT_DIR / f"{out_stem}{ext}"
            dst_txt = OUTPUT_DIR / f"{out_stem}.txt"

            shutil.copy2(src, dst_img)
            dst_txt.write_text(caption_for(cls), encoding="utf-8")

    print("LoRA dataset prepared in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
