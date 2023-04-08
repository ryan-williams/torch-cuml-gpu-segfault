#!/usr/bin/env python
import faulthandler
import sys
from subprocess import CalledProcessError

import click

faulthandler.enable()

import numpy as np


def repeat(fn, n, log=print, fail_msg='segfault', exit_early=False):
    fmt = f"%0{len(str(n))}d"
    successes, failures = 0, 0
    for i in range(n):
        ii = fmt % (i + 1)
        log(f"Iteration {ii}/{n}")
        try:
            fn()
            print(f"✅ Success (iteration {ii}/{n})")
            successes += 1
        except CalledProcessError as e:
            if e.returncode in [-11, 139]:  # segfault
                print(f"❌ Failure (iteration {ii}/{n}): exit code {e.returncode} ({fail_msg})")
                log(f"{e}")
                if exit_early:
                    raise
                failures += 1
            elif e.returncode != 1:
                raise RuntimeError(f"Unexpected returncode {e.returncode}")

    if failures:
        pct_str = "%.1f" % (failures / n * 100)
        print(f"❌ {failures}/{n} runs failed ({pct_str}%)")
        sys.exit(1)
    else:
        print(f"✅ all {successes} runs succeeded")
        sys.exit(0)


def neighbors(X):
    # ⚠️️⚠️ This (theoretically unused) import, when run before the cuml import below it, causes the pipeline to segfault
    # on ≈10% of runs. ⚠️⚠️

    # This block is adapted from scanpy.neighbors.compute_neighbors_rapids:
    # https://github.com/scverse/scanpy/blob/1.8.2/scanpy/neighbors/__init__.py#L318-L338
    from cuml.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=X.shape[1], metric='euclidean')
    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    nn.fit(X_contiguous)
    knn_indices, knn_distances = nn.kneighbors(X_contiguous)
    print(f'knn_indices ({type(knn_indices)}, {knn_indices.shape}): {knn_indices.nonzero()}, knn_distances ({type(knn_distances)}, {knn_distances.shape}): {knn_distances.nonzero()}')


quiet_opt = click.option('-q', '--quiet', is_flag=True, help='Suppress subprocess output, only print success/failure info for each iteration')
random_seed_opt = click.option('-r', '--random-seed', default=123, type=int, help='Random seed (set before `cuml.NearestNeighbors` calculation)')
shape_opt = click.option('-s', '--shape', default='10x2', help='Shape for random array passed to `cuml.NearestNeighbors`')


@click.command()
@quiet_opt
@random_seed_opt
@shape_opt
def run(*, quiet, random_seed, shape):
    log = (lambda msg: None) if quiet else print
    np.random.seed(random_seed)
    R, C = shape.split('x', 1)
    R, C = int(R), int(C)
    X = np.random.random((R, C))
    log(f"Running cuml NearestNeighbors.kneighbors on random {R}x{C} array")
    neighbors(X)
    log("Neighbors complete")


if __name__ == '__main__':
    run()
