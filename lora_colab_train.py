"""
Run this script in Google Colab T4 (or any CUDA runtime).

Expected layout in Drive:
  /content/drive/MyDrive/kyrgyz/
    lora_training/              # 100 images + 100 txt captions
    lora_output/                # created by this script

This uses kohya-style training via diffusers trainer script.
"""

import os
import subprocess
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


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "-q",
            "diffusers",
            "transformers",
            "accelerate",
            "peft",
            "datasets",
            "ftfy",
            "bitsandbytes",
        ]
    )

    train_script = "/usr/local/lib/python3.11/dist-packages/diffusers/examples/text_to_image/train_text_to_image_lora.py"
    if not Path(train_script).exists():
        raise FileNotFoundError(
            "Could not find diffusers LoRA training script. "
            "In Colab, clone diffusers repo and point train_script to examples/text_to_image/train_text_to_image_lora.py."
        )

    env = os.environ.copy()
    env["MODEL_NAME"] = MODEL_NAME
    env["DATA_DIR"] = str(DATA_DIR)
    env["OUTPUT_DIR"] = str(OUTPUT_DIR)

    cmd = [
        "accelerate",
        "launch",
        train_script,
        "--pretrained_model_name_or_path",
        MODEL_NAME,
        "--train_data_dir",
        str(DATA_DIR),
        "--resolution",
        str(RESOLUTION),
        "--center_crop",
        "--random_flip",
        "--train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "4",
        "--max_train_steps",
        str(MAX_STEPS),
        "--checkpointing_steps",
        str(SAVE_STEPS),
        "--learning_rate",
        str(LR),
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--validation_prompt",
        "kyrgyz_ornament, geometric pattern, traditional felt textile, high detail",
        "--num_validation_images",
        "4",
        "--validation_epochs",
        "1",
        "--rank",
        str(RANK),
        "--lora_alpha",
        str(ALPHA),
        "--mixed_precision",
        "fp16",
        "--output_dir",
        str(OUTPUT_DIR),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    print("Training complete. Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
