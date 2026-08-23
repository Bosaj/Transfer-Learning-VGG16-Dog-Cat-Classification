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

## [1.0.0] - 2026-08-23

### Added
- `VGG16.ipynb`: transfer learning pipeline using a frozen, ImageNet-pretrained VGG16 backbone with a custom dense classification head for dog vs. cat binary classification.
- Data augmentation via `ImageDataGenerator` (rotation, shift, shear, zoom, horizontal flip).
- Training loop (10 epochs) with reported validation accuracy of ~91.6%.
- Single-image prediction helper with visualization.
- Presentation slides and video walkthrough of the project.
