# Transfer Learning VGG16 — Dog vs. Cat Classification

Binary image classification (dog vs. cat) built on top of a VGG16 backbone pre-trained on ImageNet, using Keras/TensorFlow transfer learning.

[![CI](https://github.com/Bosaj/Transfer-Learning-VGG16-Dog-Cat-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Bosaj/Transfer-Learning-VGG16-Dog-Cat-Classification/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org/)

## Overview

This project demonstrates transfer learning by taking a VGG16 model pre-trained on ImageNet, freezing its convolutional base, and attaching a small classification head to distinguish images of dogs from cats. It was built as a hands-on machine learning exercise in applying pre-trained CNNs to a new binary classification task.

## ⚡ Try it now — live demo (`app.py`)

The notebook's fine-tuned classifier (91.6% validation accuracy) needs its trained weights and the original dogs-vs-cats training images — neither is included in this repo (both too large for git). Rather than leave the project undemonstrable, `app.py` provides a genuinely different but real transfer-learning approach: it loads the **full ImageNet-pretrained VGG16** (`include_top=True`, no fine-tuning needed) and maps its prediction to Dog/Cat using ImageNet's own class layout — dog breeds occupy indices 151-268 and domestic cat breeds occupy indices 281-285 (both verified directly against Keras's `imagenet_class_index.json`). Upload any photo and it classifies it live, with the raw top-5 ImageNet predictions shown for transparency.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deployable at [share.streamlit.io](https://share.streamlit.io) (point it at `app.py`) — no GPU or training data required, since it only runs inference on frozen ImageNet weights.

## Contents

- [`VGG16.ipynb`](VGG16.ipynb) — the full pipeline:
  1. Load `VGG16(weights='imagenet', include_top=False)` as a frozen feature extractor.
  2. Add a custom head: `Flatten -> Dense(256, relu) -> Dense(1, sigmoid)`.
  3. Compile with `Adam(learning_rate=0.0001)` and binary cross-entropy loss.
  4. Augment training images (`ImageDataGenerator`: rotation, shift, shear, zoom, horizontal flip) and rescale validation images.
  5. Train for 10 epochs on a local `trainset`/`validset` directory structure (dogs-vs-cats style dataset, not included in this repo).
  6. Evaluate loss/accuracy on the validation set and run single-image predictions with a helper `predict()` function.
- [`Transfer Learning.pptx`](<Transfer Learning.pptx>) — presentation slides summarizing the approach and results.
- [`Transfer Learning.mp4`](<Transfer Learning.mp4>) — video walkthrough.

### Results

- Reported training accuracy: **~91.6%** on the validation split (see notebook output).

## Tech Stack

- **Python**
- **TensorFlow / Keras** — `VGG16`, `Sequential`, `ImageDataGenerator`, `Adam`
- **NumPy**
- **Matplotlib** — plotting predictions
- Jupyter Notebook

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/Bosaj/Transfer-Learning-VGG16-Dog-Cat-Classification.git
cd Transfer-Learning-VGG16-Dog-Cat-Classification

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install jupyter

# 4. Provide your own dataset
# The notebook expects a dogs-vs-cats style dataset at:
#   ../Transfer_Learning/trainset/{cat,dog}/...
#   ../Transfer_Learning/validset/{cat,dog}/...
# (e.g. the Kaggle "Dogs vs. Cats" dataset), which is not included in this
# repository due to its size.

# 5. Launch Jupyter and open VGG16.ipynb
jupyter notebook
```

## Testing / CI

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR to `main`:

- Installs dependencies from `requirements.txt`
- Validates that `VGG16.ipynb` is structurally well-formed (via `nbformat`)
- Converts the notebook to a script and byte-compiles it to catch syntax errors

The CI intentionally does **not** run model training — that requires the external dogs-vs-cats dataset and is not something a hosted runner can (or should) do.

## Project Structure

```
Transfer-Learning-VGG16-Dog-Cat-Classification/
├── app.py                   # Live demo: ImageNet-VGG16 dog/cat classifier
├── VGG16.ipynb              # Main notebook: fine-tuned model, training, evaluation
├── Transfer Learning.pptx   # Presentation slides
├── Transfer Learning.mp4    # Video walkthrough
├── requirements.txt
└── .github/workflows/ci.yml
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Distributed under the MIT License — see [LICENSE](LICENSE).

## Author

**Oussama EL HADJI** — [github.com/Bosaj](https://github.com/Bosaj)


## 📊 Monitoring, Controlling, Evaluation & QA

This project includes a standardized 4-Pillar Observability and QA framework:
- **Logs & Prometheus/Grafana Monitoring**: Configured in `monitoring/` with Prometheus scraper configs and Grafana dashboards.
- **Health Controlling & Evaluation**: Liveness/readiness controllers in `monitoring/health.py` and evaluation harness in `scripts/eval_harness.py`.
- **QA & Testing**: Automated Pytest/Vitest integration and CI workflows via `.github/workflows/ci_qa_monitoring.yml`.

For complete instructions, architecture details, and commands, see [docs/MONITORING_AND_QA.md](file:///C:\Users\ROG FLOW\Desktop\Projects\Github_Projects\Transfer-Learning-VGG16-Dog-Cat-Classification\docs\MONITORING_AND_QA.md).
