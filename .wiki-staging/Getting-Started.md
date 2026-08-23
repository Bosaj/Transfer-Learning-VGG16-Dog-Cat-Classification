# Getting Started

## Prerequisites

- Python 3.10+
- TensorFlow 2.x

## Installation

```bash
git clone https://github.com/Bosaj/Transfer-Learning-VGG16-Dog-Cat-Classification.git
cd Transfer-Learning-VGG16-Dog-Cat-Classification
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter
```

Key dependencies: TensorFlow/Keras, NumPy, Matplotlib.

## Provide your own dataset

The notebook expects a dogs-vs-cats style dataset (e.g. the Kaggle "Dogs vs. Cats" dataset) at:

```
../Transfer_Learning/trainset/{cat,dog}/...
../Transfer_Learning/validset/{cat,dog}/...
```

This is not included in the repository due to its size — you need to download and arrange it yourself before training will run.

## Running the project

```bash
jupyter notebook VGG16.ipynb
```

## How CI validates the project

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every push/PR to `main`:

- Installs dependencies from `requirements.txt`.
- Validates that `VGG16.ipynb` is structurally well-formed (via `nbformat`).
- Converts the notebook to a script and byte-compiles it to catch syntax errors.

CI intentionally does not run model training — that requires the external dogs-vs-cats dataset and is not something a hosted runner can (or should) do.
