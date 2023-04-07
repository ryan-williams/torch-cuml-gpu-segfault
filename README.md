# Unused `import torch` causes nondeterministic segfault when using `cuml`

- [Reproduction steps](#repro)
  - [Create GPU instance](#create-instance)
  - [Setup GPU instance](#setup-instance)
  - [Reproduce segfault on host](#host)
  - [Reproduce segfault in Docker](#docker)
    - [1. Build Docker image](#build-docker)
    - [2. Run image repeatedly, observe occasional segfaults](#run-docker)
- [Discussion](#discussion)
  - [Removing unused `import torch` fixes it](#import)
  - [Minimizing the example](#minimizing)
  - [Python `faulthandler` not working](#faulthandler)

## Reproduction steps <a id="repro"></a>

### Create GPU instance <a id="create-instance"></a>
I've tested this on EC2 `p3.2xlarge` instances, with several AMIs:
- [`instance.tf`] is an example Terraform template for creating such a `p3.2xlarge`, using `ami-0a7de320e83dfd4ee` (["Deep Learning AMI GPU PyTorch 1.13.1 (Amazon Linux 2) 20230310"]).
- I originally reproduced this issue on several versions of Amazon's ["Deep Learning AMI (Amazon Linux 2)"][DLAMI versions]:
  - Version 57.1 (`ami-01dfbf223bd1b9835`)
  - Version 61.3 (`ami-0ac44af394b7d6689`)
  - Version 69.1 (`ami-058e8127e717f752b`)

### Setup GPU instance <a id="setup-instance"></a>
On a `p3.2xlarge` GPU instance:
```bash
git clone https://github.com/ryan-williams/torch-cuml-metaflow-gpu-segfault gpu-segfault
cd gpu-segfault
./init-conda-env.sh  # update conda, create `segfault` conda env
. ~/.bashrc          # activate `segfault` conda env
```

[`init-conda-env.sh`]:
- installs a recent Conda and configures the `libmamba-solver` (this is the quickest way to get [`environment.yml`] installed)
- creates a `segfault` Conda env from [`environment.yml`]

### Reproduce segfault on host <a id="host"></a>
[`run.py`] runs [`pipeline.py`] repeatedly (via [`entrypoint.sh`]), and shows occasional (≈10%) segfaults:
```bash
./run.py -q
# ✅ Success (iteration 01/30)
# ❌ Failure (iteration 02/30): exit code 139 (segfault)
# ✅ Success (iteration 03/30)
# ✅ Success (iteration 04/30)
# ✅ Success (iteration 05/30)
# ✅ Success (iteration 06/30)
# ✅ Success (iteration 07/30)
# ✅ Success (iteration 08/30)
# ✅ Success (iteration 09/30)
# ❌ Failure (iteration 10/30): exit code 139 (segfault)
# ✅ Success (iteration 11/30)
# ✅ Success (iteration 12/30)
# ✅ Success (iteration 13/30)
# ✅ Success (iteration 14/30)
# ✅ Success (iteration 15/30)
# ✅ Success (iteration 16/30)
# ✅ Success (iteration 17/30)
# ✅ Success (iteration 18/30)
# ✅ Success (iteration 19/30)
# ✅ Success (iteration 20/30)
# ❌ Failure (iteration 21/30): exit code 139 (segfault)
# ✅ Success (iteration 22/30)
# ✅ Success (iteration 23/30)
# ✅ Success (iteration 24/30)
# ✅ Success (iteration 25/30)
# ✅ Success (iteration 26/30)
# ✅ Success (iteration 27/30)
# ✅ Success (iteration 28/30)
# ✅ Success (iteration 29/30)
# ✅ Success (iteration 30/30)
❌ 3/30 runs failed (10.0%)
```

### Reproduce segfault in Docker <a id="docker"></a>
The same behavior can be observed in a Docker image, built from this repo and run using `--runtime nvidia`:

#### 1. Build Docker image <a id="build-docker"></a>
```bash
img=segfault
docker build -t$img .
```

See [Dockerfile](Dockerfile).

#### 2. Run image repeatedly, observe occasional segfaults <a id="run-docker"></a>
In this case, [`run.py`] repeatedly runs the Docker `$img` built above:
```bash
./run.py -d $img -q
# ✅ Success (iteration 01/30)
# ✅ Success (iteration 02/30)
# ✅ Success (iteration 03/30)
# ❌ Failure (iteration 04/30): exit code 139 (segfault in Docker)
# ✅ Success (iteration 05/30)
# ✅ Success (iteration 06/30)
# ✅ Success (iteration 07/30)
# ✅ Success (iteration 08/30)
# ✅ Success (iteration 09/30)
# ✅ Success (iteration 10/30)
# ✅ Success (iteration 11/30)
# ✅ Success (iteration 12/30)
# ✅ Success (iteration 13/30)
# ✅ Success (iteration 14/30)
# ✅ Success (iteration 15/30)
# ✅ Success (iteration 16/30)
# ✅ Success (iteration 17/30)
# ✅ Success (iteration 18/30)
# ✅ Success (iteration 19/30)
# ✅ Success (iteration 20/30)
# ❌ Failure (iteration 21/30): exit code 139 (segfault in Docker)
# ✅ Success (iteration 22/30)
# ❌ Failure (iteration 23/30): exit code 139 (segfault in Docker)
# ✅ Success (iteration 24/30)
# ✅ Success (iteration 25/30)
# ✅ Success (iteration 26/30)
# ✅ Success (iteration 27/30)
# ✅ Success (iteration 28/30)
# ✅ Success (iteration 29/30)
# ✅ Success (iteration 30/30)
```

## Discussion <a id="discussion"></a>
[The "pipeline" here](pipeline.py) has one non-empty step, `start`, which:
- Generates a 10x2 random matrix, `X`
- Instantiates a [`cuml.neighbors.NearestNeighbors`] and fits it to `X`
- Calls [`kneighbors`] on `X`

### Removing unused `import torch` fixes it <a id="import"></a>
Something about [this `import torch`][`import torch`] is side-effectful, and creates a condition where some cleanup process seg-faults while Metaflow is cleaning up the step. Presumably the data structures in question are instantiated by `cuml` (probably `cudf` DataFrames); if [the `nn.kneighbors` call](pipeline.py#L21-L22) is commented out, the segfault also goes away.

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

### Minimizing the example <a id="minimizing"></a>
This is as minimal of a repro as I've found for this issue, which initially manifested during a call to [`scanpy.preprocessing.neighbors`] on larger, private data, during a (larger) [`Metaflow`] pipeline run.

Here I've isolated the specific Metaflow step (by pre-populating some data in [the `.metaflow-example` directory](.metaflow-example/Pipeline)) to avoid extra subprocess calls Metaflow would otherwise make en route to hitting the segfault.

I've also tried to reduce dependencies and update versions of what remains (see [`environment.yml`]). My original context is tied to Python 3.9 and CUDA 11.6, hence the [`cuml==22.06.01&lsqb;build=cuda11_py39*&rsqb;`](environment.yml#L8) pin. I originally hit the issue on Torch 1.12.1, but it persists with Torch 1.13.1 here.

### Python `faulthandler` not working <a id="faulthandler"></a>
I've tried to enable a more detailed stack trace from the segfault in a few places ([Dockerfile#L4](Dockerfile#L4), [entrypoint.sh#L4](entrypoint.sh#L4), [pipeline.py#L7](pipeline.py#L7)), per [these instructions][segfault debug article], but have so far been unable to get any more info about where it is occurring.


[`scanpy.preprocessing.neighbors`]: https://github.com/scverse/scanpy/blob/1.8.2/scanpy/neighbors/__init__.py#L52
[`scanpy.neighbors.compute_neighbors_rapids`]: https://github.com/scverse/scanpy/blob/1.8.2/scanpy/neighbors/__init__.py#L318
[`environment.yml`]: environment.yml
[`cuml.neighbors.NearestNeighbors`]: https://github.com/rapidsai/cuml/blob/v22.06.01/python/cuml/neighbors/nearest_neighbors.pyx#L153
[`kneighbors`]: https://github.com/rapidsai/cuml/blob/v22.06.01/python/cuml/neighbors/nearest_neighbors.pyx#L482
[`Metaflow`]: https://metaflow.org/
[`import torch`]: pipeline.py#L14
[pipeline.py]: pipeline.py
[segfault debug article]: https://blog.richard.do/2018/03/18/how-to-debug-segmentation-fault-in-python/
[`run.py`]: run.py
[`pipeline.py`]: pipeline.py
[DLAMI versions]: https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html
[`instance.tf`]: instance.tf
[`init-conda-env.sh`]: init-conda-env.sh
[`entrypoint.sh`]: entrypoint.sh

["Deep Learning AMI GPU PyTorch 1.13.1 (Amazon Linux 2) 20230310"]: https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-1-13-amazon-linux-2/
