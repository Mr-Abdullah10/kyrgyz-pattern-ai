"""
Run in Colab after LoRA training.
Generates 50 images and saves with class-prefixed names.
"""

from pathlib import Path

import torch
from diffusers import DiffusionPipeline


BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_DIR = Path("/content/drive/MyDrive/kyrgyz/lora_output")
OUT_DIR = Path("/content/drive/MyDrive/kyrgyz/generated_gallery")
SEED = 42


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.load_lora_weights(str(LORA_DIR))
    pipe = pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(SEED)

    for cls, count in COUNTS.items():
        prompts = PROMPTS[cls]
        for i in range(count):
            prompt = prompts[i % len(prompts)]
            img = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
            out_name = f"{cls}_{i+1:02d}.jpg"
            img.save(OUT_DIR / out_name, quality=95)
            print("saved", out_name)


if __name__ == "__main__":
    main()
