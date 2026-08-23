# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MIT `LICENSE`.
- `requirements.txt` pinning TensorFlow/NumPy/Matplotlib.
- GitHub Actions CI workflow validating notebook integrity and syntax.
- Rewritten `README.md` with tech stack, structure, and getting-started instructions.
- `app.py`: a live, deployable Streamlit demo. The notebook's fine-tuned weights and training images aren't in this repo (too large for git), so this uses the full ImageNet-pretrained VGG16 directly and maps its prediction to Dog/Cat via ImageNet's own class index ranges (151-268 for dog breeds, 281-285 for domestic cats - verified against Keras's `imagenet_class_index.json`). A different technique from the notebook's fine-tuned head, but genuine transfer learning requiring no training data, documented as such in the README.

## [1.0.0] - 2026-08-23

### Added
- `VGG16.ipynb`: transfer learning pipeline using a frozen, ImageNet-pretrained VGG16 backbone with a custom dense classification head for dog vs. cat binary classification.
- Data augmentation via `ImageDataGenerator` (rotation, shift, shear, zoom, horizontal flip).
- Training loop (10 epochs) with reported validation accuracy of ~91.6%.
- Single-image prediction helper with visualization.
- Presentation slides and video walkthrough of the project.
