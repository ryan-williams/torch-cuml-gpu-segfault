# Unused `import torch` causes nondeterministic segfault when using `cuml`

- [Reproduction steps](#repro)
  - [Create P3-class GPU instance](#create-instance)
  - [Setup GPU instance](#setup-instance)
  - [Reproduce segfault on host](#host)
  - [Reproduce segfault in Docker](#docker)
- [Discussion](#discussion)
  - [Removing unused `import torch` fixes it](#import)
  - [Minimizing the example](#minimizing)
  - [Python `faulthandler` not working](#faulthandler)

## Reproduction steps <a id="repro"></a>

I've tested this on EC2 `p3.2xlarge` instances, with [an NVIDIA V100 GPU][p3 instances], with several AMIs:
- [`ami-03fce349214ac583f`]: Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20221226
- [`ami-0a7de320e83dfd4ee`]: Deep Learning AMI GPU PyTorch 1.13.1 (Amazon Linux 2) 20230310
- `ami-003f25e6e2d2db8f1`: NVIDIA GPU-Optimized AMI 22.06.0-676eed8d-dcf5-4784-87d7-0de463205c17 (marketplace image, "subscribe" for free [here](https://aws.amazon.com/marketplace/server/procurement?productId=676eed8d-dcf5-4784-87d7-0de463205c17))
- Several versions of Amazon's ["Deep Learning AMI (Amazon Linux 2)"]:
  - Version 57.1 (`ami-01dfbf223bd1b9835`)
  - Version 61.3 (`ami-0ac44af394b7d6689`)
  - Version 69.1 (`ami-058e8127e717f752b`)

### Create P3-class GPU instance <a id="create-instance"></a>
Any "P3" family instance seems to exhibit the behavior, but:
- [cdk/] contains [CDK] scripts for booting an instance with a configurable AMI
  - Uses [`ami-0a7de320e83dfd4ee`] ("Deep Learning AMI GPU PyTorch 1.13.1 (Amazon Linux 2) 20230310") by default
  - Runs [`init-conda-env.sh`] on instance boot (make sure you [wait until that's done][cdk#async], when you first log in!)
- [`instance.tf`] is an example Terraform template for doing similar
  - Doesn't initialize node, see instructions below:

You may need to request a quota increase for P-class instance vCPUs, if you've never launched one:

```bash
aws service-quotas request-service-quota-increase \
    --service-code ec2 \
    --quota-code L-417A185B \
    --desired-value 8
```

[This page](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-417A185B) should also work.

Other GPU instance types may also exhibit the issue, I've just tested on `p3.2xlarge`s.

### Setup GPU instance <a id="setup-instance"></a>
On a `p3.2xlarge` GPU instance:
```bash
git clone https://github.com/ryan-williams/torch-cuml-gpu-segfault gpu-segfault
cd gpu-segfault
time ./init-conda-env.sh  # update conda, create `segfault` conda env; can take ≈15mins!
. ~/.bashrc               # activate `segfault` conda env
```

The [`init-conda-env.sh`] script:
- installs a recent Conda and configures the `libmamba-solver` (this is the quickest way to get [`environment.yml`] installed)
- creates a `segfault` Conda env from [`environment.yml`]

The [cdk/] scripts run the commands above asynchronously during instance boot; [use `tail -f /var/log/cloud-init-output.log` to see when it's done][cdk#async], if you use those scripts.

### Reproduce segfault on host <a id="host"></a>
[`run.py`] runs [`neighbors.py`] repeatedly, and shows occasional (≈10%) segfaults:
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

#### 1. Build Docker image
```bash
img=segfault
docker build -t$img .
```

See [Dockerfile](Dockerfile).

#### 2. Run image repeatedly, observe occasional segfaults
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
[`neighbors.py`] performs the following steps:
- Generate a 10x2 random matrix, `X`
- Instantiate a [`cuml.neighbors.NearestNeighbors`], fit it to `X`
- Call [`kneighbors`] on `X`

### Removing unused `import torch` fixes it <a id="import"></a>
Something about [this `import torch`][`import torch`] is side-effectful, and causes a segmentation fault (apparently while the Python process is exiting).

Perhaps there is a "use after free" problem with some data structures instantiated by `cuml` (probably `cudf` DataFrames)?

If [the `nn.kneighbors` call](neighbors.py#L22) is commented out, the segfault goes away.

If you remove [the unused `import torch` in neighbors.py][`import torch`], the segfault also goes away:
```python
def neighbors(X):
    # ⚠️️⚠️ This (theoretically unused) import, when executed before the cuml import below it, leads to a segfault 
    # (seemingly during Python process cleanup) on ≈10% of runs. ⚠️⚠️
    import torch
    from cuml.neighbors import NearestNeighbors
```

```bash
perl -pi -e 's/import torch/# import torch/' neighbors.py
python run.py  # ✅ now everything succeeds!
```

### Minimizing the example <a id="minimizing"></a>
This is as minimal of a repro as I've found for this issue (which initially manifested in a [Metaflow] pipeline, during a call to [`scanpy.preprocessing.neighbors`], on larger, proprietary data).

I've tried to reduce dependencies and update versions of what remains (see [`environment.yml`]). I originally hit the issue with CUDA 11.6, `cuml==22.06.01`, and Torch 1.12.1, but the repro persists on CUDA 11.7, `cuml==22.12.00`, and Torch 1.13.1.

### Python `faulthandler` not working <a id="faulthandler"></a>
I've tried to enable a more detailed stack trace from the segfault in a few places ([Dockerfile#L4](Dockerfile#L13), [neighbors.py#L7](neighbors.py#L6)), per [these instructions][segfault debug article], but have so far been unable to get any more info about where the segfault is occurring.


[`scanpy.preprocessing.neighbors`]: https://github.com/scverse/scanpy/blob/1.8.2/scanpy/neighbors/__init__.py#L52
[`scanpy.neighbors.compute_neighbors_rapids`]: https://github.com/scverse/scanpy/blob/1.8.2/scanpy/neighbors/__init__.py#L318
[`environment.yml`]: environment.yml
[`cuml.neighbors.NearestNeighbors`]: https://github.com/rapidsai/cuml/blob/v22.06.01/python/cuml/neighbors/nearest_neighbors.pyx#L153
[`kneighbors`]: https://github.com/rapidsai/cuml/blob/v22.06.01/python/cuml/neighbors/nearest_neighbors.pyx#L482
[`import torch`]: neighbors.py#L14
[segfault debug article]: https://blog.richard.do/2018/03/18/how-to-debug-segmentation-fault-in-python/
[`run.py`]: run.py
[`instance.tf`]: instance.tf
[`init-conda-env.sh`]: init-conda-env.sh

["Deep Learning AMI (Amazon Linux 2)"]: https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html
[AWS Deep Learning AMI (Amazon Linux 2)]: https://aws.amazon.com/releasenotes/aws-deep-learning-ami-amazon-linux-2/
[`ami-0a7de320e83dfd4ee`]: https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-1-13-amazon-linux-2/
[`ami-03fce349214ac583f`]: https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-1-13-ubuntu-20-04/

[p3 instances]: https://aws.amazon.com/ec2/instance-types/p3/
[CDK]: https://aws.amazon.com/cdk/
[cdk/]: cdk
[`neighbors.py`]: neighbors.py
[Metaflow]: https://metaflow.org/
[cdk#async]: cdk/README.md#async
