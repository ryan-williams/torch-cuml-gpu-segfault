# Unused `import torch` causes nondeterministic segfault when using `cuml`

- [Reproduction steps](#repro)
  - [0. Setup GPU environment](#setup)
  - [1. Build Docker image](#build)
  - [2. Run image repeatedly, observe occasional segfaults](#run)
- [Removing unused `import torch` fixes it](#import)
- [Discussion](#import)
  - [Minimizing the example](#minimizing)

## Reproduction steps <a id="repro"></a>
### 0. Setup GPU environment <a id="setup"></a>
TODO:
- EC2 `p3.2xlarge` instance
- Docker + nvidia runtime

### 1. Build Docker image <a id="build"></a>
```bash
img=segfault
docker build -t$img .
```

See [Dockerfile](Dockerfile). It has an `ENTRYPOINT` that invokes [pipeline.py], which imports `torch` and then runs a `cuml` nearest-neighbors function.

### 2. Run image repeatedly, observe occasional segfaults <a id="run"></a>
```bash
n=30
for i in `seq $n`; do
    ii="`printf %02d $i`"
    echo "Iteration $ii/$n"
    docker run -it --rm $img
    rv=$?
    [ $rv -eq 0 ] \
    && echo "✅ Success (iteration $ii/$n)" \
    || echo "❌ Failure (iteration $ii/$n); exit code $rv"
done | grep iteration
# ✅ Success (iteration 01/30)
# ✅ Success (iteration 02/30)
# ✅ Success (iteration 03/30)
# ✅ Success (iteration 04/30)
# ✅ Success (iteration 05/30)
# ❌ Failure (iteration 06/30); exit code 139
# ✅ Success (iteration 07/30)
# ✅ Success (iteration 08/30)
# ✅ Success (iteration 09/30)
# ✅ Success (iteration 10/30)
# ❌ Failure (iteration 11/30); exit code 139
# ✅ Success (iteration 12/30)
# ✅ Success (iteration 13/30)
# ✅ Success (iteration 14/30)
# ✅ Success (iteration 15/30)
# ✅ Success (iteration 16/30)
# ❌ Failure (iteration 17/30); exit code 139
# ✅ Success (iteration 18/30)
# ❌ Failure (iteration 19/30); exit code 139
# ✅ Success (iteration 20/30)
# ✅ Success (iteration 21/30)
# ✅ Success (iteration 22/30)
# ❌ Failure (iteration 23/30); exit code 139
# ❌ Failure (iteration 24/30); exit code 139
# ✅ Success (iteration 25/30)
# ❌ Failure (iteration 26/30); exit code 139
# ✅ Success (iteration 27/30)
# ✅ Success (iteration 28/30)
# ✅ Success (iteration 29/30)
# ✅ Success (iteration 30/30)
```

The [`run.py`](run.py) script does something similar, but exits on the first failure:
```bash
run.py
# Usage: run.py [OPTIONS] IMG
#
#   Repeatedly run a docker image, exiting on the first error
#
# Options:
#   -n, --num-repetitions INTEGER
#   --help                         Show this message and exit.
```

## Removing unused `import torch` fixes it <a id="import"></a>
If you remove [the unused `import torch` in pipeline.py][`import torch`], the segfaults go away:
```python
def neighbors(X):
    # ⚠️️⚠️ This (theoretically unused) import, when run before the cuml import below it, causes the pipeline to segfault
    # on ≈10% of runs. ⚠️⚠️
    import torch
    from cuml.neighbors import NearestNeighbors
```

```bash
perl -pi -e 's/import torch/# import torch/' pipeline.py
python run.py  # ✅ now everything succeeds!
```

## Discussion <a id="import"></a>
[The "pipeline" here](pipeline.py) has one non-empty step, `start`, which:
- Generates a 10x2 random matrix, `X`
- Instantiates a [`cuml.neighbors.NearestNeighbors`] and fits it to `X`
- Calls [`kneighbors`] on `X`

Something about [this `import torch`][`import torch`] is side-effectful, and creates a condition where some cleanup process seg-faults while Metaflow is cleaning up the step. Presumably the data structures in question are instantiated by `cuml` (probably `cudf` DataFrames); if [the `nn.kneighbors` call](pipeline.py#L21-L22) is commented out, the segfault also goes away.

### Minimizing the example <a id="minimizing"></a>
This is as minimal of a repro as I've found for this issue, which initially manifested during a call to [`scanpy.preprocessing.neighbors`] on larger, private data, during a (larger) [`Metaflow`] pipeline run.

Here I've isolated the specific Metaflow step (by [pre-populating some data in the Docker image's `.metaflow` directory](Dockerfile#L39); see [metaflow/](metaflow)), to avoid extra subprocess calls Metaflow would otherwise make.

I've also tried to reduce dependencies and update versions of what remains (see [`environment.yml`]). My original context is tied to Python 3.9 and CUDA 11.6, hence the [`cuml==22.06.01&lsqb;build=cuda11_py39*&rsqb;`](environment.yml#L8) pin. I originally hit the issue on Torch 1.12.1, but it persists with Torch 1.13.1 as here.


[`scanpy.preprocessing.neighbors`]: https://github.com/scverse/scanpy/blob/1.8.2/scanpy/neighbors/__init__.py#L52
[`scanpy.neighbors.compute_neighbors_rapids`]: https://github.com/scverse/scanpy/blob/1.8.2/scanpy/neighbors/__init__.py#L318
[`environment.yml`]: environment.yml
[`cuml.neighbors.NearestNeighbors`]: https://github.com/rapidsai/cuml/blob/v22.06.01/python/cuml/neighbors/nearest_neighbors.pyx#L153
[`kneighbors`]: https://github.com/rapidsai/cuml/blob/v22.06.01/python/cuml/neighbors/nearest_neighbors.pyx#L482
[`Metaflow`]: https://metaflow.org/
[`import torch`]: pipeline.py#L14
[pipeline.py]: pipeline.py
