import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv_models
import open_clip
import faiss
import numpy as np
import pandas as pd
import timm
from PIL import Image
from pathlib import Path

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Kyrgyz Pattern AI",
    page_icon="🏔️",
    layout="wide"
)

CLASSES       = ["animal", "geometric", "symbolic"]
RETRIEVAL_DIR = Path("retrieval")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Available classifier models ───────────────────────────────
AVAILABLE_MODELS = {}
for name, path in [
    ("ResNet50",       Path("checkpoints/resnet50_final.pth")),
    ("MobileNetV2",    Path("checkpoints/mobilenet_final.pth")),
    ("EfficientNet-B0", Path("checkpoints/kyrgyz_classifier_final.pth")),
]:
    if path.exists():
        AVAILABLE_MODELS[name] = path

# ── LOAD MODELS — cached so they only load once ───────────────
@st.cache_resource
def load_classifier(arch_name: str, ckpt_path: str):
    """Build and load a classifier by architecture name."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    if arch_name == "ResNet50":
        model = tv_models.resnet50(weights=None)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(in_feat, 512),
            nn.ReLU(inplace=True), nn.Dropout(p=0.3),
            nn.Linear(512, 3),
        )
    elif arch_name == "MobileNetV2":
        model = tv_models.mobilenet_v2(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(in_feat, 256),
            nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(256, 3),
        )
    else:  # EfficientNet-B0 (legacy)
        model = timm.create_model(
            "efficientnet_b0", pretrained=False,
            drop_rate=0.4, drop_path_rate=0.2
        )
        in_feat = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(in_feat, 3)
        )

    model.load_state_dict(state)
    model.to(DEVICE).eval()

    # Return model + metadata from checkpoint
    meta = {
        "accuracy": ckpt.get("test_accuracy", "N/A"),
        "f1": ckpt.get("weighted_f1", "N/A"),
    }
    return model, meta

@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    return model.to(DEVICE).eval(), preprocess

@st.cache_resource
def load_retrieval():
    index    = faiss.read_index(str(RETRIEVAL_DIR / "faiss.index"))
    metadata = pd.read_csv(RETRIEVAL_DIR / "metadata.csv")
    embs     = np.load(RETRIEVAL_DIR / "embeddings.npy").astype("float32")
    return index, metadata, embs

# ── PREPROCESSING FOR CLASSIFIER ─────────────────────────────
clf_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# ── CLASSIFIER PREDICTION ─────────────────────────────────────
def classify_image(pil_img, clf_model):
    tensor = clf_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = clf_model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    top_idx  = int(np.argmax(probs))
    return CLASSES[top_idx], float(probs[top_idx]), probs

# ── CLIP EMBEDDING FOR ONE IMAGE ──────────────────────────────
def get_clip_embedding(pil_img, clip_model, preprocess):
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")

# ── FAISS RETRIEVAL ───────────────────────────────────────────
def retrieve_similar(embedding, index, metadata, top_k=3):
    scores, indices = index.search(embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = metadata.iloc[idx]
        results.append({
            "filename"    : row["filename"],
            "class"       : row["class"],
            "path"        : row["path"],
            "pattern_name": row.get("pattern_name", ""),
            "meaning"     : row.get("meaning", ""),
            "similarity"  : float(score),
        })
    return results

# ── CONFIDENCE BAR COLOR ──────────────────────────────────────
def confidence_color(conf):
    if conf >= 0.75: return "green"
    if conf >= 0.55: return "orange"
    return "red"

# ══════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════

st.title("Kyrgyz Traditional Pattern AI")
st.caption("Upload a pattern image to classify it and discover its cultural meaning.")

# ── Model Selection ───────────────────────────────────────────
if AVAILABLE_MODELS:
    model_names = list(AVAILABLE_MODELS.keys())
    selected_model = st.sidebar.selectbox(
        "🧠 Classification Model",
        model_names,
        index=0,
        help="Choose which trained model to use for pattern classification",
    )
    st.sidebar.markdown("---")
else:
    selected_model = None
    st.sidebar.error("No trained model checkpoints found in checkpoints/ folder.")

# Load all models
with st.spinner("Loading models..."):
    if selected_model:
        clf_model, clf_meta = load_classifier(
            selected_model, str(AVAILABLE_MODELS[selected_model])
        )
        # Show model info in sidebar
        st.sidebar.markdown(f"**Active Model:** {selected_model}")
        if isinstance(clf_meta.get('accuracy'), float):
            st.sidebar.markdown(f"Test Accuracy: `{clf_meta['accuracy']*100:.1f}%`")
            st.sidebar.markdown(f"Weighted F1: `{clf_meta['f1']:.4f}`")
    else:
        clf_model, clf_meta = None, {}
    clip_model, preprocess = load_clip()
    faiss_index, metadata, _ = load_retrieval()

tab1, tab2, tab3 = st.tabs(["Recognize Pattern", "Generated Gallery", "Browse Dataset"])

# ── TAB 1: RECOGNIZE ─────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Upload a Kyrgyz pattern image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(pil_img, caption="Uploaded image", use_container_width=True)

        with col2:
            with st.spinner("Classifying..."):
                pred_class, confidence, all_probs = classify_image(pil_img, clf_model)

            class_labels = {
                "animal"   : "Animal-Inspired (Zoomorphic)",
                "geometric": "Geometric",
                "symbolic" : "Symbolic / Floral",
            }
            st.subheader(f"Predicted: {class_labels[pred_class]}")

            color = confidence_color(confidence)
            st.markdown(
                f"**Confidence:** "
                f"<span style='color:{color}; font-size:18px'>"
                f"{confidence*100:.1f}%</span>",
                unsafe_allow_html=True
            )

            st.markdown("**Class probabilities:**")
            for cls, prob in zip(CLASSES, all_probs):
                st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

            if confidence < 0.60:
                st.warning(
                    "Low confidence — this pattern may overlap multiple categories. "
                    "Check the similar patterns below for context."
                )

        st.markdown("---")
        st.subheader("Most Similar Patterns in Dataset")

        with st.spinner("Searching dataset..."):
            emb     = get_clip_embedding(pil_img, clip_model, preprocess)
            similar = retrieve_similar(emb, faiss_index, metadata, top_k=3)

        cols = st.columns(3)
        for col, result in zip(cols, similar):
            with col:
                try:
                    sim_img = Image.open(result["path"]).convert("RGB")
                    st.image(sim_img, use_container_width=True)
                except:
                    st.info("Image not available")
                st.markdown(f"**Class:** {result['class']}")
                st.markdown(f"**Similarity:** {result['similarity']*100:.1f}%")
                if result["pattern_name"]:
                    st.markdown(f"**Name:** {result['pattern_name']}")
                if result["meaning"]:
                    st.markdown(f"**Meaning:** {result['meaning']}")

# ── TAB 2: GENERATE ───────────────────────────────────────────
with tab2:
    st.subheader("🎨 Generate Kyrgyz Pattern")
    st.caption(
        "Type a prompt and the AI will generate a brand-new Kyrgyz-style "
        "pattern image in real time using cloud-based Stable Diffusion."
    )

    GENERATED_DIR = Path("generated_gallery")

    # Initialize generation history in session state
    if "generation_history" not in st.session_state:
        st.session_state.generation_history = []

    # ── Prompt templates ──────────────────────────────────────
    PROMPT_TEMPLATES = {
        "Select a template...": "",
        "Geometric — red and black diamond grid shyrdak": "kyrgyz_ornament, geometric shyrdak pattern, red and black, diamond grid, mirror symmetry, traditional felt textile, high detail",
        "Geometric — traditional mosaic felt carpet pattern": "kyrgyz_ornament, geometric mosaic carpet, traditional Kyrgyz felt pattern, repeating rhombus tiles, handcrafted textile, high detail",
        "Geometric — symmetric Kyrgyz border ornament": "kyrgyz_ornament, symmetric border ornament, geometric zigzag edges, traditional Kyrgyz carpet trim, high detail",
        "Animal — kochkor muyuz ram horn spiral motif": "kyrgyz_ornament, kochkor muyuz ram horn spirals, zoomorphic ornament, red and black felt textile, curved S-shapes, high detail",
        "Animal — bugu muyuz deer horn zoomorphic pattern": "kyrgyz_ornament, bugu muyuz deer horn motif, zoomorphic Kyrgyz pattern, curved ornamental elements, traditional carpet design, high detail",
        "Animal — traditional zoomorphic felt carpet": "kyrgyz_ornament, animal-inspired nomadic ornament, swirling horn elements, zoomorphic pattern, handcrafted felt carpet style, high detail",
        "Symbolic — tunduk solar circle motif blue and white": "kyrgyz_ornament, tunduk sun symbol, radial floral ornament, blue and white, traditional Kyrgyz felt textile, circular composition, high detail",
        "Symbolic — floral embroidery tush kiyiz style": "kyrgyz_ornament, floral tush kiyiz embroidery, symbolic Kyrgyz pattern, vine and flower motifs, decorative textile art, high detail",
        "Symbolic — Kyrgyz rose and butterfly wall hanging": "kyrgyz_ornament, rose and butterfly symbolic pattern, Kyrgyz wall hanging motif, colorful floral ornament, detailed textile artwork, high detail",
    }

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Choose a template**")
        selected_template = st.selectbox(
            "Prompt template",
            list(PROMPT_TEMPLATES.keys()),
            label_visibility="collapsed"
        )

        st.markdown("**Or type your own prompt**")
        custom_prompt = st.text_area(
            "Custom prompt",
            placeholder="e.g. geometric diamond pattern with red and black colors on felt carpet",
            height=80,
            label_visibility="collapsed"
        )

        generate_btn = st.button(
            "🎨 Generate Pattern",
            type="primary",
            use_container_width=True
        )

    with col_right:
        st.markdown("**How it works**")
        st.info(
            "Your prompt is automatically enhanced with Kyrgyz pattern "
            "styling context and sent to a cloud AI model for generation.\n\n"
            "**Tip:** Be descriptive! Include colors, motif types, and style.\n\n"
            "Example: `geometric diamond pattern, red and black, shyrdak carpet`"
        )

        # Show backend status
        import os
        stability_key = os.getenv("STABILITY_API_KEY", "").strip()
        hf_token = os.getenv("HF_TOKEN", "").strip()

        if stability_key:
            st.success("✅ **Stability AI** — highest quality")
        elif hf_token:
            st.success("✅ **HuggingFace** — good quality (free)")
        else:
            st.warning("🌐 **Pollinations.ai** — free, no key needed")
            with st.expander("Want better quality?"):
                st.markdown(
                    "Set API keys in a `.env` file for higher quality:\n\n"
                    "```\n"
                    "STABILITY_API_KEY=your_key_here\n"
                    "HF_TOKEN=your_token_here\n"
                    "```\n\n"
                    "Get keys at:\n"
                    "- [Stability AI](https://platform.stability.ai/account/keys)\n"
                    "- [HuggingFace](https://huggingface.co/settings/tokens) (free)"
                )

    # ── Generation logic ──────────────────────────────────────
    if generate_btn:
        # Determine the prompt to use
        active_prompt = ""
        if custom_prompt.strip():
            active_prompt = custom_prompt.strip()
        elif selected_template and selected_template != "Select a template...":
            active_prompt = PROMPT_TEMPLATES[selected_template]

        if not active_prompt:
            st.warning("⚠️ Please select a template or type a custom prompt.")
        else:
            st.markdown("---")
            with st.spinner("🎨 Generating pattern... This may take 15–60 seconds."):
                try:
                    from generate_service import generate_image

                    image, filepath, backend, enhanced_prompt = generate_image(
                        active_prompt
                    )

                    # Display the result
                    st.success(f"✅ Pattern generated via **{backend}**!")

                    img_col, info_col = st.columns([2, 1])
                    with img_col:
                        st.image(
                            image,
                            caption="AI-Generated Kyrgyz Pattern",
                            use_container_width=True
                        )
                    with info_col:
                        st.markdown("**Your prompt:**")
                        st.code(active_prompt, language=None)
                        st.markdown("**Enhanced prompt sent to AI:**")
                        st.code(enhanced_prompt, language=None)
                        st.markdown(f"**Backend:** {backend}")
                        st.markdown(f"**Saved:** `{Path(filepath).name}`")

                    # Add to history
                    st.session_state.generation_history.insert(0, {
                        "image_path": filepath,
                        "prompt": active_prompt,
                        "backend": backend,
                    })

                except Exception as e:
                    st.error(f"❌ Generation failed: {e}")
                    st.info(
                        "**Troubleshooting:**\n"
                        "- Check your internet connection\n"
                        "- If using an API key, verify it's valid\n"
                        "- Try a simpler prompt\n"
                        "- The free service may be temporarily busy — try again"
                    )

    # ── Show generation history ───────────────────────────────
    if st.session_state.generation_history:
        st.markdown("---")
        st.markdown("**Recent generations**")
        hist_cols = st.columns(min(4, len(st.session_state.generation_history)))
        for i, entry in enumerate(st.session_state.generation_history[:4]):
            with hist_cols[i]:
                try:
                    hist_img = Image.open(entry["image_path"]).convert("RGB")
                    st.image(hist_img, use_container_width=True)
                    st.caption(f'{entry["prompt"][:40]}...')
                except:
                    st.info("Image unavailable")

    # ── Show full gallery below ───────────────────────────────
    st.markdown("---")
    st.markdown("**All generated images**")

    gen_filter = st.selectbox(
        "Filter by class",
        ["All", "animal", "geometric", "symbolic"],
        key="gen_filter"
    )

    if GENERATED_DIR.exists():
        if gen_filter == "All":
            all_gen = sorted(
                [p for p in GENERATED_DIR.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                 and p.name != ".gitkeep"],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        else:
            all_gen = sorted(
                [p for p in GENERATED_DIR.glob(f"{gen_filter.lower()}_*")
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            )
            # Also include newly generated ones with "gen_" prefix
            all_gen += sorted(
                [p for p in GENERATED_DIR.glob(f"gen_{gen_filter.lower()}_*")
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            )

        st.caption(f"{len(all_gen)} images")
        gcols = st.columns(4)
        for i, p in enumerate(all_gen):
            with gcols[i % 4]:
                try:
                    st.image(Image.open(p).convert("RGB"), use_container_width=True)
                    st.caption(p.stem.replace("_", " "))
                except:
                    pass
    else:
        st.warning("Generated gallery folder not found.")

# ── TAB 3: DATASET GALLERY ────────────────────────────────────
with tab3:
    st.subheader("Dataset Gallery")

    filter_class = st.selectbox(
        "Filter by class",
        ["All", "animal", "geometric", "symbolic"],
        key="dataset_filter"
    )

    df = metadata.copy()
    if filter_class != "All":
        df = df[df["class"] == filter_class]

    st.caption(f"Showing {len(df)} images")

    cols = st.columns(4)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 4]:
            try:
                img = Image.open(row["path"]).convert("RGB")
                st.image(img, use_container_width=True)
                st.caption(f"{row['class']}")
            except:
                pass