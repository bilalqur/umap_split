# umap-stratified-split
![Python](https://img.shields.io/badge/Python-3.8+-blue)
> Stratified dataset splitting via UMAP‑based pseudo‑labels  
> A drop‑in alternative to `torch.utils.data.random_split` that ensures each split preserves the manifold structure of your data.

## 📖 Overview

When you randomly split a dataset, you risk concentrating your validation set in a narrow region of feature space—especially in small or highly clustered datasets.  
`umap-stratified-split` embeds your entire dataset with UMAP (with fixed seed), clusters the embedding into “pseudo‑labels,” then performs stratified splitting to guarantee that each subset visits **all** manifold regions evenly.

This approach is grounded in the assumption that your dataset lies on a meaningful low-dimensional manifold and benefits from a uniform, structure-preserving sampling across this space.

---

## 🧠 Theoretical Motivation

This method builds on the foundation laid by the UMAP algorithm:

> **Uniform Manifold Approximation and Projection (UMAP)** is a general-purpose nonlinear dimension reduction technique. Its core idea is to construct a graph-based fuzzy topological representation of data, then optimize a low-dimensional embedding that preserves local neighborhood structure.

UMAP relies on the following assumptions:

1. **The data is uniformly distributed on a Riemannian manifold**
2. **The Riemannian metric is locally constant** (or approximately so)
3. **The manifold is locally connected**

These assumptions allow UMAP to build a reliable low-dimensional embedding that captures meaningful cluster and density information.

By clustering the UMAP embedding and stratifying across those clusters, this package provides a principled way to sample validation and training sets that are *representative of the entire data manifold*.

---

## ✅ Advantages of this Approach

- **Manifold-aware validation**: Ensures your validation set covers the same regions as your training set
- **Less validation bias**: Avoids selecting validation samples from narrow regions of the space
- **General-purpose**: Works for any data type (images, time series, embeddings, tabular, etc.)
- **Drop-in replacement**: Mimics the `random_split` API from PyTorch
- **Fully unsupervised**: Uses the geometry of the data, no true labels required
- **Reproducible**: UMAP seed and clustering make the split deterministic

---

## ⚠️ When to Use / Limitations

This method works best when:

- Your data lies on a structured manifold (e.g. clustered, continuous trajectories)
- The standard random split leads to class imbalance or structural bias
- You don’t have true labels, but want stratified-like splits based on learned structure

Avoid using this method if:

- Your data is completely uniform or already well-distributed
- You are splitting into a *test set* (risk of data leakage via unsupervised embedding)
- Your feature extractor or UMAP embedding fails to capture meaningful structure

---

## 📚 References

- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv preprint [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
- Official UMAP implementation: https://github.com/lmcinnes/umap

---

## 🔧 Installation

```bash
# install latest from PyPI
pip install umap-stratified-split

# or install editable from GitHub
pip install git+https://github.com/bilalqur/umap-stratified-split.git#egg=umap-stratified-split

# or locally in editable/develop mode
git clone https://github.com/bilalqur/umap-stratified-split.git
cd umap-stratified-split
pip install -e .
```

---
