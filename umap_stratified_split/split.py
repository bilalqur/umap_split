import math
from typing import Sequence, Optional, Callable, List, Tuple
from torch.utils.data import Dataset, Subset
import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def umap_stratified_split(
    dataset: Dataset,
    lengths: Sequence[int],
    *,
    feature_extractor: Optional[Callable[[any], np.ndarray]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    n_clusters_factor: int = 10,
    random_state: Optional[int] = None,
) -> List[Subset]:
    """
    Split `dataset` into len(lengths) subsets of sizes `lengths`,
    such that each subset has approximately the same distribution in UMAP space.

    Args:
      dataset: any torch Dataset supporting __len__ and __getitem__.
      lengths: list of subset sizes (must sum to len(dataset)).
      feature_extractor: fn(x) -> 1D numpy array. If None, assumes x itself is numeric array.
      n_neighbors, min_dist, n_components: UMAP params.
      n_clusters_factor: creates n_clusters = n_splits * n_clusters_factor (capped at N).
      random_state: for UMAP, KMeans, and train_test_split.

    Returns:
      A list of torch.utils.data.Subset objects, in the same order as `lengths`.
    """
    N = len(dataset)
    if sum(lengths) != N:
        raise ValueError(f"Sum of lengths {sum(lengths)} ≠ dataset size {N}")

    # 1) Build feature matrix
    X = []
    for idx in range(N):
        item = dataset[idx]
        vec = feature_extractor(item) if feature_extractor else np.asarray(item)
        X.append(vec.ravel())
    X = np.stack(X, axis=0)

    # 2) Embed via UMAP
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    emb = reducer.fit_transform(X)

    # 3) Cluster embedding to get pseudo‑labels
    n_splits = len(lengths)
    n_clusters = min(N, n_splits * n_clusters_factor)
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(emb)

    # 4) Sequential stratified splits
    all_indices = np.arange(N)
    all_labels  = labels
    subsets: List[np.ndarray] = []

    remaining_lengths = list(lengths)
    for i, Li in enumerate(lengths[:-1]):
        prop = Li / sum(remaining_lengths)
        # stratify current pool by label
        idx_train, idx_rem = train_test_split(
            all_indices,
            stratify=all_labels,
            train_size=prop,
            random_state=random_state,
        )
        subsets.append(idx_train)
        # update pools
        all_indices = idx_rem
        all_labels  = all_labels[np.isin(np.arange(N), idx_rem)]
        remaining_lengths.pop(0)

    # Last chunk takes everything left
    subsets.append(all_indices)

    # Wrap in Subset
    return [Subset(dataset, idx.tolist()) for idx in subsets]
