# Methodology

## Backbone

`VGG16(weights='imagenet', include_top=False)` is loaded as a frozen feature extractor — its convolutional layers keep their ImageNet-trained weights and are not updated during training.

## Classification head

A small dense head is attached on top of the frozen backbone:

```
Flatten -> Dense(256, relu) -> Dense(1, sigmoid)
```

The single sigmoid output gives a binary probability (cat vs. dog).

## Training setup

- **Optimizer**: `Adam(learning_rate=0.0001)`
- **Loss**: binary cross-entropy
- **Data augmentation**: `ImageDataGenerator` applies rotation, width/height shift, shear, zoom, and horizontal flip to training images; validation images are only rescaled (no augmentation).
- **Epochs**: 10, on a local `trainset`/`validset` directory structure.

## Results

Training accuracy climbed quickly from 66% in epoch 1 to over 84% by epoch 2, then fluctuated in the high-80s to low-90s range through epoch 10, with validation accuracy peaking at **92.2%** (epoch 9) and the notebook reporting an overall figure of **~91.6%**. This fluctuation across epochs (rather than smooth convergence) is typical for a small, frozen-backbone model with a relatively high augmentation load and no learning-rate schedule.

## Inference

A helper `predict()` function loads a single image, preprocesses it the same way as training/validation data, and displays the image alongside the model's predicted class using Matplotlib.
