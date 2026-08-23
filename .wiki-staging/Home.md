# Transfer Learning VGG16 — Dog vs. Cat Classification

Binary image classification (dog vs. cat) built on top of a VGG16 backbone pre-trained on ImageNet, using Keras/TensorFlow transfer learning.

The project freezes VGG16's convolutional base and attaches a small dense classification head, trains it with augmented images for 10 epochs, and reaches roughly **91.6% validation accuracy**. It was built as a hands-on exercise in applying pre-trained CNNs to a new binary classification task, and includes presentation slides and a video walkthrough alongside the notebook.

## Quick Links

- [Getting Started](Getting-Started) — installation, dataset setup, and how to run the notebook
- [Methodology](Methodology) — model architecture, training setup, and results
- [FAQ](FAQ) — common questions about the dataset and the approach
