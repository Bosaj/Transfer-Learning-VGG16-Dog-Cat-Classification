# FAQ

**Why freeze the VGG16 backbone instead of fine-tuning it?**
Freezing the convolutional base is the standard "feature extraction" flavor of transfer learning — it reuses the general-purpose visual features VGG16 already learned from ImageNet without risking overfitting the relatively small dogs-vs-cats training set by updating millions of pretrained weights.

**Where do I get the dataset?**
It's not included in the repository due to size. Use a dogs-vs-cats style dataset (e.g. Kaggle's "Dogs vs. Cats") and arrange it into `trainset/{cat,dog}/` and `validset/{cat,dog}/` folders as described in [Getting Started](Getting-Started).

**Why does validation accuracy bounce around instead of steadily improving?**
With only 10 epochs, a fixed learning rate, and heavy data augmentation, some epoch-to-epoch variance in validation accuracy is expected — the model is still learning overall (see the epoch-by-epoch numbers in [Methodology](Methodology)), but the smaller dataset size for this exercise doesn't smooth it into a perfectly monotonic curve.

**Can this be extended to more than two classes?**
Yes — the general approach (frozen backbone + custom head) extends directly to multi-class problems by swapping the final `Dense(1, sigmoid)` for `Dense(n_classes, softmax)` and using categorical cross-entropy loss.

**Why doesn't CI actually train the model?**
Training needs the external image dataset (not checked into the repo for size reasons) and would be too slow/resource-intensive for a shared CI runner. CI instead checks that the notebook is valid and that converting it to a script byte-compiles cleanly.

**What are the `.pptx` and `.mp4` files for?**
They're supplementary material — presentation slides and a video walkthrough summarizing the approach and results, useful for a quick overview without reading the full notebook.
