#!/usr/bin/env python

import numpy as np

def neighbors(X):
    # ⚠️️⚠️ This (theoretically unused) import, when run before the cuml import below it, causes the pipeline to segfault
    # on ≈10% of runs. ⚠️⚠️
    import torch

    # This block is adapted from scanpy.neighbors.compute_neighbors_rapids:
    # https://github.com/scverse/scanpy/blob/1.8.2/scanpy/neighbors/__init__.py#L318-L338
    from cuml.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=X.shape[1], metric='euclidean')
    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    nn.fit(X_contiguous)
    knn_indices, knn_distances = nn.kneighbors(X_contiguous)
    print(f'knn_indices ({type(knn_indices)}, {knn_indices.shape}): {knn_indices.nonzero()}, knn_distances ({type(knn_distances)}, {knn_distances.shape}): {knn_distances.nonzero()}')


def run(random_seed=123, shape='10x2'):
    np.random.seed(random_seed)
    R, C = shape.split('x', 1)
    R, C = int(R), int(C)
    X = np.random.random((R, C))
    print(f"Running cuml NearestNeighbors.kneighbors on random {R}x{C} array")
    neighbors(X)
    print("Neighbors complete")


if __name__ == '__main__':
    run()
