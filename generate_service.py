"""
Cloud-based image generation for Kyrgyz Pattern AI.

Since the local GPU (MX250, 2GB VRAM) cannot run Stable Diffusion,
this service routes generation requests to cloud APIs.

Backend priority (tries in order, uses first that works):
  1. Stability AI   — best quality, needs STABILITY_API_KEY env var
  2. HuggingFace     — good quality, needs HF_TOKEN env var (free tier)
  3. Together AI     — good quality, needs TOGETHER_API_KEY env var (free tier)
  4. Pollinations.ai — decent quality, NO key needed, always works (with retry)

Usage:
    from generate_service import generate_image
    img, filepath, backend, enhanced = generate_image("geometric diamond pattern")
"""

import base64
import io
import os
import random
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────
GENERATED_DIR = Path(__file__).resolve().parent / "generated_gallery"
GENERATED_DIR.mkdir(exist_ok=True)

# Kyrgyz-specific prompt styling injected into every generation
KYRGYZ_STYLE_SUFFIX = (
    ", traditional Kyrgyz felt textile ornament, Central Asian nomadic art, "
    "handcrafted pattern, shyrdak carpet style, vibrant colors, "
    "high detail, symmetrical design, cultural heritage motif, ornamental art"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, text, watermark, signature, modern, digital art, "
    "3d render, photograph, person, face, hands, anime, cartoon, logo"
)

# Retry settings for rate-limited APIs
MAX_RETRIES = 4
BASE_DELAY = 5  # seconds


# ── PROMPT ENHANCEMENT ────────────────────────────────────────
def enhance_prompt(user_prompt: str) -> str:
    """Add Kyrgyz pattern styling context to the user's raw prompt."""
    prompt = user_prompt.strip()
    # Ensure trigger word is present
    if "kyrgyz" not in prompt.lower():
        prompt = f"kyrgyz_ornament, {prompt}"
    # Add style suffix if not already detailed
    if "high detail" not in prompt.lower() and "detailed" not in prompt.lower():
        prompt += KYRGYZ_STYLE_SUFFIX
    return prompt


# ── BACKEND 1: STABILITY AI ──────────────────────────────────
def generate_stability(prompt: str, api_key: str) -> Image.Image:
    """Generate via Stability AI REST API (paid, highest quality)."""
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "text_prompts": [
            {"text": prompt, "weight": 1.0},
            {"text": NEGATIVE_PROMPT, "weight": -1.0},
        ],
        "cfg_scale": 7,
        "height": 512,
        "width": 512,
        "steps": 30,
        "samples": 1,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 200:
        data = resp.json()
        img_b64 = data["artifacts"][0]["base64"]
        return Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    raise RuntimeError(f"Stability AI error {resp.status_code}: {resp.text[:300]}")


# ── BACKEND 2: HUGGINGFACE INFERENCE API ──────────────────────
def generate_huggingface(
    prompt: str,
    hf_token: str,
    model: str = "",
) -> Image.Image:
    """Generate via HuggingFace Inference API (free tier available)."""
    if not model:
        model = os.getenv(
            "HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
        )
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
        },
    }

    resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
    if resp.status_code == 200:
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    raise RuntimeError(f"HuggingFace error {resp.status_code}: {resp.text[:300]}")


# ── BACKEND 3: TOGETHER AI ───────────────────────────────────
def generate_together(prompt: str, api_key: str) -> Image.Image:
    """Generate via Together AI API (free tier: 1000 requests/day)."""
    url = "https://api.together.xyz/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "black-forest-labs/FLUX.1-schnell-Free",
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "steps": 4,
        "n": 1,
        "response_format": "b64_json",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 200:
        data = resp.json()
        img_b64 = data["data"][0]["b64_json"]
        return Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    raise RuntimeError(f"Together AI error {resp.status_code}: {resp.text[:300]}")


# ── BACKEND 4: POLLINATIONS.AI (free, no key) ────────────────
def generate_pollinations(prompt: str) -> Image.Image:
    """Generate via Pollinations.ai — completely free, no API key needed.
    Includes retry logic for 429 rate-limit errors."""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        seed = random.randint(1, 999999)
        encoded = urllib.parse.quote(prompt)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width=512&height=512&nologo=true&seed={seed}"
        )

        try:
            print(f"[Pollinations] Attempt {attempt}/{MAX_RETRIES} (seed={seed})")
            resp = requests.get(url, timeout=150)

            if resp.status_code == 200 and len(resp.content) > 1000:
                return Image.open(io.BytesIO(resp.content)).convert("RGB")

            if resp.status_code == 429:
                wait = BASE_DELAY * attempt + random.uniform(0, 3)
                print(f"[Pollinations] Rate limited (429). Waiting {wait:.1f}s...")
                last_error = f"Rate limited (attempt {attempt})"
                time.sleep(wait)
                continue

            last_error = f"HTTP {resp.status_code}, body size={len(resp.content)}"
            # For non-429 errors, still retry but with shorter wait
            time.sleep(BASE_DELAY)

        except requests.exceptions.Timeout:
            last_error = f"Timeout on attempt {attempt}"
            print(f"[Pollinations] Timeout on attempt {attempt}")
            time.sleep(BASE_DELAY)
        except Exception as e:
            last_error = str(e)
            time.sleep(BASE_DELAY)

    raise RuntimeError(f"Pollinations failed after {MAX_RETRIES} attempts: {last_error}")


# ── MAIN ENTRY POINT ─────────────────────────────────────────
def generate_image(user_prompt: str) -> tuple:
    """
    Generate an image from a text prompt using the best available cloud backend.

    Returns:
        (PIL.Image, filepath: str, backend_name: str, enhanced_prompt: str)
    """
    enhanced = enhance_prompt(user_prompt)

    stability_key = os.getenv("STABILITY_API_KEY", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    together_key = os.getenv("TOGETHER_API_KEY", "").strip()

    image = None
    backend = ""
    errors = []

    # Try backends in quality order
    if stability_key:
        try:
            image = generate_stability(enhanced, stability_key)
            backend = "Stability AI"
        except Exception as e:
            errors.append(f"Stability AI: {e}")
            print(f"[generate] Stability AI failed: {e}")

    if image is None and hf_token:
        try:
            image = generate_huggingface(enhanced, hf_token)
            backend = "HuggingFace"
        except Exception as e:
            errors.append(f"HuggingFace: {e}")
            print(f"[generate] HuggingFace failed: {e}")

    if image is None and together_key:
        try:
            image = generate_together(enhanced, together_key)
            backend = "Together AI"
        except Exception as e:
            errors.append(f"Together AI: {e}")
            print(f"[generate] Together AI failed: {e}")

    if image is None:
        try:
            image = generate_pollinations(enhanced)
            backend = "Pollinations.ai"
        except Exception as e:
            errors.append(f"Pollinations: {e}")
            error_summary = " | ".join(errors)
            raise RuntimeError(
                f"All generation backends failed.\n{error_summary}"
            )

    # Save the generated image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Detect class keyword for filename prefix
    prompt_lower = user_prompt.lower()
    if any(w in prompt_lower for w in ["animal", "horn", "kochkor", "bugu", "zoomorphic"]):
        prefix = "animal"
    elif any(w in prompt_lower for w in ["geometric", "diamond", "shyrdak", "mosaic", "grid"]):
        prefix = "geometric"
    elif any(w in prompt_lower for w in ["symbolic", "tunduk", "solar", "floral", "flower"]):
        prefix = "symbolic"
    else:
        prefix = "pattern"

    filename = f"gen_{prefix}_{timestamp}.jpg"
    filepath = GENERATED_DIR / filename
    image.save(filepath, quality=95)

    return image, str(filepath), backend, enhanced

