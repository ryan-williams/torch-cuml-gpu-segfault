import faulthandler

import numpy as np

from metaflow import FlowSpec, Parameter, resources, step

faulthandler.enable()


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


class Pipeline(FlowSpec):
    """Simple Metaflow pipeline exercising the segfaulting `import torch` → `cuml.NearestNeighbors` code path.

    I've so far not reproduced the segfault outside of Metaflow; I think something about its step-cleanup process is
    either causing or leaving time for a memory error related to `cuml` data-structures.
    """
    random_seed = Parameter('random_seed', type=int, default=123)
    shape = Parameter('shape', type=str, default='10x2')

    @resources(memory=200_000, cpu=1, gpu=1)
    @step
    def start(self):
        run(random_seed=self.random_seed, shape=self.shape)
        print('finished run, calling self.next')
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    Pipeline()
