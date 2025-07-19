import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
from umap_stratified_split.split import umap_stratified_split
import matplotlib.pyplot as plt
from umap import UMAP


# Load CIFAR-10 with flattening
class CIFAR10Vec(datasets.CIFAR10):
    def __getitem__(self, idx):
        img, lbl = super().__getitem__(idx)
        return np.array(img).reshape(-1) / 255.0, lbl  # Flatten to 3072-dim

# Download and prepare dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = CIFAR10Vec(root="data", train=True, download=True, transform=transform)

# Define feature extractor
def feature_extractor(item):
    vec, label = item
    return vec  # just the flattened image

# Perform stratified split
train_set, val_set = umap_stratified_split(
    dataset,
    lengths=(45000, 5000),
    feature_extractor=feature_extractor,
    n_neighbors=15,
    n_components=2,
    n_clusters_factor=10,
    random_state=42
)

print(f"Train size: {len(train_set)}\tVal size: {len(val_set)}")

# Plot class distribution
from collections import Counter

def count_labels(subset):
    return Counter([dataset[i][1] for i in subset.indices])

train_counts = count_labels(train_set)
val_counts   = count_labels(val_set)

labels = range(10)
train_vals = [train_counts[l] for l in labels]
val_vals   = [val_counts[l] for l in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, train_vals, width, label='Train')
plt.bar(x + width/2, val_vals, width, label='Val')
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution: Train vs Val (CIFAR-10)")
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
plt.show()

# UMAP visualization
print("Generating UMAP visualization of split distribution...")

all_indices = train_set.indices + val_set.indices
X = np.array([feature_extractor(dataset[i]) for i in all_indices])
split_labels = np.array(['train'] * len(train_set.indices) + ['val'] * len(val_set.indices))

reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X)

plt.figure(figsize=(8, 6))
for label, color in zip(['train', 'val'], ['blue', 'orange']):
    mask = split_labels == label
    plt.scatter(embedding[mask, 0], embedding[mask, 1], s=5, c=color, label=label, alpha=0.5)
plt.title("UMAP Projection of CIFAR-10\nColored by Split")
plt.legend()
plt.tight_layout()
plt.show()
