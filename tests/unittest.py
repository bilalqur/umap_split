import numpy as np
import pytest
from torch.utils.data import Dataset, Subset

from umap_stratified_split.split import umap_stratified_split


class SyntheticDataset(Dataset):
    """
    A simple synthetic dataset with two clusters:
    - First half of points around (0, 0)
    - Second half around (10, 10)
    Each item is a 2D numpy array.
    """
    def __init__(self, n_samples=100, cluster_std=0.5, random_state=None):
        rng = np.random.RandomState(random_state)
        half = n_samples // 2
        cluster1 = rng.randn(half, 2) * cluster_std + np.array([0, 0])
        cluster2 = rng.randn(n_samples - half, 2) * cluster_std + np.array([10, 10])
        self.data = np.vstack([cluster1, cluster2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return raw vector
        return self.data[idx]


def test_split_lengths():
    ds = SyntheticDataset(n_samples=100, random_state=0)
    splits = umap_stratified_split(ds, lengths=(80, 20), random_state=42)
    # Ensure we get two subsets
    assert isinstance(splits, list)
    assert len(splits) == 2
    # Check lengths
    assert len(splits[0]) == 80
    assert len(splits[1]) == 20
    # Combined indices cover all
    all_indices = set(splits[0].indices + splits[1].indices)
    assert all_indices == set(range(len(ds)))


def _count_cluster(subset):
    # count how many samples come from each original cluster
    data = np.array([subset.dataset[i] for i in subset.indices])
    # cluster by threshold x<5
    c1 = np.sum(data[:, 0] < 5)
    c2 = np.sum(data[:, 0] >= 5)
    return c1, c2


def test_stratified_proportions():
    ds = SyntheticDataset(n_samples=100, random_state=0)
    train, val = umap_stratified_split(ds, lengths=(80, 20), random_state=42)

    # Count cluster membership
    t1, t2 = _count_cluster(train)
    v1, v2 = _count_cluster(val)

    # Expect approximate ratios: 80% of each cluster in train, 20% in val
    assert pytest.approx(t1 / 50, rel=0.1) == 0.8
    assert pytest.approx(t2 / 50, rel=0.1) == 0.8
    assert pytest.approx(v1 / 50, rel=0.1) == 0.2
    assert pytest.approx(v2 / 50, rel=0.1) == 0.2


def test_deterministic_behavior():
    ds = SyntheticDataset(n_samples=100, random_state=0)
    # Two splits with same seed
    split1 = umap_stratified_split(ds, lengths=(80, 20), random_state=123)
    split2 = umap_stratified_split(ds, lengths=(80, 20), random_state=123)
    # The indices order might differ in each subset, so sort before comparing
    assert sorted(split1[0].indices) == sorted(split2[0].indices)
    assert sorted(split1[1].indices) == sorted(split2[1].indices)

    # Different seed should differ
    split3 = umap_stratified_split(ds, lengths=(80, 20), random_state=456)
    assert sorted(split1[0].indices) != sorted(split3[0].indices) or \
           sorted(split1[1].indices) != sorted(split3[1].indices)