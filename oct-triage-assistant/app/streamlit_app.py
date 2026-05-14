"""
Streamlit demo for educational OCT triage-style predictions.

UX goals:
- Clear **non-diagnostic** language and a visible disclaimer.
- Show predicted label and confidence (max softmax probability).
- **Escalation hints**: low confidence → suggest human review; CNV/DME → suggest clinical review
  (still not stating a medical diagnosis).
"""

from __future__ import annotations

import streamlit as st
import torch
from PIL import Image

from src import config
from src.model import build_model, load_trained_weights
from src.predict import predict_image
from src.transforms import get_val_transforms


DISCLAIMER = (
    "This tool is for educational purposes only and is not a medical device or diagnostic system. "
    "It does not provide a diagnosis. Do not use for clinical decision-making."
)


@st.cache_resource
def load_model_bundle():
    ckpt_path = config.CHECKPOINT_PATH
    if not ckpt_path.is_file():
        return None
    device = torch.device(config.DEVICE)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes: list[str] = list(ckpt.get("classes", list(config.CLASSES)))
    tfm = get_val_transforms(config.IMG_SIZE)
    model = build_model(num_classes=len(classes), pretrained=False)
    load_trained_weights(model, ckpt_path, map_location=device)
    model.to(device)
    return model, tfm, device, classes


def escalation_notes(label: str, confidence: float) -> list[str]:
    notes: list[str] = []
    if confidence < config.LOW_CONFIDENCE_THRESHOLD:
        notes.append("Low confidence. Human expert review recommended.")
    if label in {"CNV", "DME"}:
        notes.append("Potentially urgent retinal finding. Clinical review recommended.")
    return notes


def main() -> None:
    st.set_page_config(page_title="OCT Triage Assistant (Educational)", layout="centered")
    st.title("Image-Based Retinal OCT Triage Assistant")
    st.caption("Educational / portfolio use — not for patient care.")

    st.error(DISCLAIMER)

    bundle = load_model_bundle()
    if bundle is None:
        st.warning(
            f"No model checkpoint found at `{config.CHECKPOINT_PATH}`. "
            "Train the model first: `python -m src.train` from the project root."
        )
        return

    model, tfm, device, classes = bundle

    st.markdown(
        """
**What this app does**

Upload a single OCT-like image. The model returns an **educational triage-style class**
among: **CNV**, **DME**, **DRUSEN**, **NORMAL**, plus a **confidence score** (highest predicted probability).
This is **not** a statement of disease presence or absence in a medico-legal sense.
        """.strip()
    )

    uploaded = st.file_uploader("Upload an OCT image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

    if uploaded is None:
        st.info("Upload an image to see a prediction.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - Streamlit demo: show friendly error
        st.error(f"Could not read the uploaded file as an image: {exc}")
        return

    st.image(image, caption="Uploaded image (preview)", use_container_width=True)

    label, confidence, probs = predict_image(model, image, tfm, device, classes)

    st.subheader("Educational model output")
    st.write(f"**Predicted triage-style class:** `{label}`")
    st.write(f"**Confidence (max softmax probability):** `{confidence:.3f}`")

    with st.expander("All class probabilities (softmax)"):
        for c, p in sorted(probs.items(), key=lambda x: -x[1]):
            st.write(f"**{c}** — {p:.3f}")
            st.progress(min(max(p, 0.0), 1.0))

    notes = escalation_notes(label, confidence)
    if notes:
        st.subheader("Safety / escalation hints")
        for n in notes:
            st.warning(n)

    st.markdown(
        """
**How to interpret responsibly**

- Treat results as a **learning artifact** tied to your specific training data and splits.
- **Do not** claim FDA/CE clearance or clinical validation unless you actually have it.
- For research prototypes, report metrics on clearly described test data (see `src/evaluate.py`).
        """.strip()
    )


if __name__ == "__main__":
    main()
