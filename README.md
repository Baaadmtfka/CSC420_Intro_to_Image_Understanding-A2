# CSC420 – Assignment 2: Image Pyramids, Backpropagation & Transfer Learning

Individual assignment for University of Toronto's CSC420 (Introduction to Image Understanding), covering image pyramids, neural network fundamentals (backprop, activation functions, CNN parameter/FLOP counts), and a PyTorch transfer-learning study on two dog-image datasets ("DBI" and "SDD") under domain shift.

## Layout

- **`a2-writeup.tex` / `.pdf`** — the report (Part I theory + Part II implementation results), with figures `q3a.jpg`, `IIa/IIb.png`, `III.a/III.b.png`, `IV.a/IV.b/IV.c.png`, `V.png`.
- **`assignment2/a2_implementation.ipynb`** — the implementation notebook, plus `README.txt` describing the dataset setup.
- **`Dataset/DBI.zip`, `Dataset/SDD.zip`** — the two dog-image dataset subsets, tracked via Git LFS (see `.gitattributes`). Unzipping them locally reproduces `Dataset/Original/` and `Dataset/Ver1/`, which aren't tracked in git.

## Part I: Theoretical problems (`a2-writeup.tex`)

- **Q1 — Image pyramids**: derives the reconstruction formula recovering the original image from a Laplacian pyramid plus the coarsest Gaussian level.
- **Q2 — Activation functions**: shows a multi-layer network with purely linear activations collapses to a single affine map.
- **Q3 — Backpropagation**: forward pass and a full manual gradient derivation (`∂L/∂w3`) through a small computational graph.
- **Q4 — CNN FLOPs/parameters**: computes FLOP counts for a conv + pooling layer (with and without bias), and total trainable parameters for a LeNet-style architecture (61,706 params).
- **Q6/Q7 — Logistic vs. tanh activations**: derives both activation functions' derivatives in terms of their own outputs, and compares their properties for use in hidden vs. output layers.

## Part II: Implementation (`assignment2/a2_implementation.ipynb`, PyTorch)

- **Task I — Inspection**: compares the DBI and SDD dog-image subsets; SDD has more pose/background variability and occluding objects than the more centered, cleaner DBI images.
- **Task II — `SimpleCNN`**: a small conv net (4 conv layers + FC head) trained on DBI, with and without dropout, to show dropout's regularizing effect on test-accuracy stability.
- **Task III — ResNet18 from scratch (`pretrained=False`)**: trained on DBI, then evaluated zero-shot on SDD to quantify the accuracy drop from domain shift (DBI 0.446 vs. SDD 0.333 test accuracy).
- **Task IV — Fine-tuning ImageNet-pretrained ResNet18 / ResNet34 / ResNeXt50_32x4d**: same DBI→SDD generalization comparison; ResNeXt50 generalizes best across both datasets.
- **Task V — `DatasetClassifier`**: a fine-tuned ResNeXt50_32x4d that classifies which of the two datasets (DBI vs. SDD) an image came from (0.815 test accuracy).
- **Tasks VI/VII — Discussion**: strategies for improving SDD performance under varying data-access assumptions (augmentation, partial-label fine-tuning, pseudo-labeling), and general domain-shift mitigation (dropout, augmentation, pretrained fine-tuning).

## Running

The notebook was run on Google Colab with a T4 High-RAM GPU (up to 14.7 GB GPU RAM used). To run locally instead of mounting Google Drive, unzip `Dataset/DBI.zip` and `Dataset/SDD.zip` into `Dataset/` and point `ImageFolder(...)` at the local paths (already toggled in the notebook — see the commented-out Drive-mount cells).

## Author

Zixuan Zeng
