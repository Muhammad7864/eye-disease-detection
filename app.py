import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eye Disease Predictor",
    page_icon="👁️",
    layout="centered",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 720px; }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
    }
    .disclaimer {
        font-size: 12px;
        color: #888;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "efficientnetb0_eye_disease.h5"
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
IMG_SIZE = (224, 224)

BADGE_COLORS = {
    "Normal":               ("🟢", "#EAF3DE", "#27500A"),
    "Cataract":             ("🟡", "#FAEEDA", "#633806"),
    "Glaucoma":             ("🔴", "#FAECE7", "#4A1B0C"),
    "Diabetic Retinopathy": ("🟣", "#EEEDFE", "#26215C"),
}

BAR_COLORS = {
    "Normal":               "#639922",
    "Cataract":             "#BA7517",
    "Glaucoma":             "#D85A30",
    "Diabetic Retinopathy": "#7F77DD",
}

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("## 👁️ Eye Disease Predictor")
st.markdown(
    "Upload a **fundus (retinal) photograph** to screen for "
    "**Cataract**, **Glaucoma**, **Diabetic Retinopathy**, or **Normal**."
)
st.divider()

model = load_model()
if model is None:
    st.error(
        f"Model file `{MODEL_PATH}` not found. "
        "Place it in the same directory as `app.py` and restart."
    )
    st.stop()

uploaded = st.file_uploader(
    "Upload fundus image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded fundus image", use_container_width=True)
    st.divider()

    with st.spinner("Analyzing image…"):
        arr = preprocess(image)
        preds = model.predict(arr, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    top_label = CLASS_NAMES[top_idx]
    top_conf = float(preds[top_idx]) * 100

    emoji, bg, fg = BADGE_COLORS[top_label]

    st.markdown(f"### Prediction")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            f"<span style='background:{bg};color:{fg};padding:6px 16px;"
            f"border-radius:999px;font-weight:600;font-size:16px'>"
            f"{emoji} {top_label}</span>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Confidence", f"{top_conf:.1f}%")

    st.markdown("#### Probability scores")
    for label, score in zip(CLASS_NAMES, preds):
        pct = float(score) * 100
        color = BAR_COLORS[label]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
            f"<span style='width:160px;font-size:14px;color:#555'>{label}</span>"
            f"<div style='flex:1;background:#f0f0f0;border-radius:4px;height:10px'>"
            f"<div style='width:{pct:.1f}%;background:{color};height:10px;border-radius:4px'></div>"
            f"</div>"
            f"<span style='width:44px;text-align:right;font-size:13px;color:#555'>{pct:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<div class='disclaimer'>⚠️ This tool is for research and educational use only. "
        "It is not a substitute for professional ophthalmological diagnosis. "
        "Always consult a qualified eye care professional.</div>",
        unsafe_allow_html=True,
    )
