# SAEVision - Sparse Autoencoders for ViT Activations

This repository bundles several sparse autoencoders (SAEs) and helper utilities for exploring monosemantic features in the CLS-token activations of vision transformers.

Implemented autoencoders:

* **ReLUSAE** - classic sparse autoencoder with ReLU sparsity.
* **TopKSAE** - enforces exactly *k* active features per input.
* **JumpReLUSAE** - JumpReLU non-linearity with learnable thresholds.
* **OrthogonalSAE** - competition-aware training with orthogonality and temporal consistency terms.
