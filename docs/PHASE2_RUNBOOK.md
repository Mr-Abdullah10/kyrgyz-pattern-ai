# Phase 2 LoRA Runbook

## 1) Prepare LoRA training folder
- Run locally:
  - `py -3 prepare_lora_dataset.py`
- Output: `lora_training/` with 100 images + 100 `.txt` captions.
- Validate quickly:
  - every caption starts with `kyrgyz_ornament`
  - class balance: `34 geometric`, `33 animal`, `33 symbolic`

## 2) Train on Colab T4
- Upload `lora_training/` to Google Drive: `MyDrive/kyrgyz/lora_training/`.
- In Colab:
  - mount drive
  - run `lora_colab_train.py`
- Key settings already configured:
  - base model: `runwayml/stable-diffusion-v1-5`
  - LoRA rank/alpha: `16/16`
  - resolution: `512`
  - steps: `3000`
  - checkpoint interval: `500`
  - precision: `fp16`

## 3) Generate 50-image gallery
- In Colab (after training), run:
  - `generate_lora_gallery.py`
- Output: `MyDrive/kyrgyz/generated_gallery/`
  - `17 geometric`, `17 animal`, `16 symbolic`

## 4) Copy gallery to API project
- Copy generated images into:
  - `generated_gallery/` in this project root
- API endpoint:
  - `GET /generate/gallery?cls=all|animal|geometric|symbolic`

## 5) Backend smoke test
- Start API:
  - `uvicorn api:app --reload --host 0.0.0.0 --port 8000`
- Verify:
  - `GET /health`
  - `GET /classes`
  - `GET /gallery?cls=geometric&page=1`
  - `GET /generate/gallery?cls=animal`
