"""Interactive dog vs. cat classifier using VGG16 transfer learning.

Honesty note: the original notebook (VGG16.ipynb) fine-tuned a custom
classification head on top of a frozen VGG16 base, trained on a private
Kaggle Dogs-vs-Cats split (3,000 train / 1,000 validation images,
reaching 91.6% validation accuracy) - but neither the trained weights
nor the training images are included in this repository (too large for
git), so that exact model cannot be reloaded here.

This app instead uses the full VGG16 network pretrained on ImageNet
(include_top=True, the standard 1000-class ImageNet classifier) and
maps its prediction to Dog/Cat using ImageNet's own class index ranges:
dog breeds occupy indices 151-268 and domestic cat breeds occupy
indices 281-285 (both contiguous blocks in the standard ImageNet-1k
label ordering). This is genuine transfer learning - it reuses VGG16's
already-learned ImageNet features and classifier with no additional
training required - just a different (and honestly disclosed) strategy
than fine-tuning a custom head on the missing dataset.
"""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input

# ImageNet-1k class index ranges (standard ordering).
DOG_INDEX_RANGE = range(151, 269)
CAT_INDEX_RANGE = range(281, 286)


@st.cache_resource
def load_model():
    return VGG16(weights="imagenet", include_top=True)


def classify(model, image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img).astype("float32"), axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr, verbose=0)[0]
    top5_idx = np.argsort(preds)[::-1][:5]

    dog_score = float(sum(preds[i] for i in DOG_INDEX_RANGE))
    cat_score = float(sum(preds[i] for i in CAT_INDEX_RANGE))

    if dog_score > cat_score and dog_score > 0.05:
        verdict = "Dog"
    elif cat_score > dog_score and cat_score > 0.05:
        verdict = "Cat"
    else:
        verdict = "Neither (not confidently a dog or cat)"

    readable = decode_predictions(preds.reshape(1, -1), top=5)[0]
    return verdict, dog_score, cat_score, readable


st.title("Dog vs. Cat Classifier (VGG16 Transfer Learning)")
st.caption(
    "Uses ImageNet-pretrained VGG16 directly, mapping its 1000-class "
    "prediction to Dog/Cat via ImageNet's own breed-class index ranges "
    "- see the module docstring in app.py for why this differs from the "
    "original notebook's fine-tuned approach."
)

model = load_model()

uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=300)

    with st.spinner("Classifying..."):
        verdict, dog_score, cat_score, top5 = classify(model, image)

    if verdict == "Dog":
        st.success(f"🐶 **Dog** (dog-breed confidence: {dog_score:.1%})")
    elif verdict == "Cat":
        st.success(f"🐱 **Cat** (cat-breed confidence: {cat_score:.1%})")
    else:
        st.warning(f"🤷 {verdict} (dog: {dog_score:.1%}, cat: {cat_score:.1%})")

    with st.expander("Raw top-5 ImageNet predictions"):
        for _, label, score in top5:
            st.write(f"{label.replace('_', ' ')}: {score:.1%}")
else:
    st.info("Upload a dog or cat photo to classify it.")
