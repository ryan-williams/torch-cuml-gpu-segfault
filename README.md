# Unused `import torch` causes nondeterministic segfault when using [`cuml`]
Importing [PyTorch] is side-effectful, and causes a segfault after `cuml.NearestNeighbors` execution (≈10% of the time, apparently during Python process cleanup).

- [Reproduction steps](#repro)
  - [Create P3-class GPU instance](#create-instance)
  - [Setup GPU instance](#setup-instance)
  - [Reproduce segfault on host](#host)
  - [Reproduce segfault in Docker](#docker)
- [Discussion](#discussion)
  - [Removing unused `import torch` fixes it](#import)
  - [Minimizing the example](#minimizing)
  - [Python `faulthandler` not working](#faulthandler)
  - [Side-effectful `import torch` breaks with RAPIDS](#torch-vs-rapids)
  - [Sensitivity calculations](#sensitivity)

## Reproduction steps <a id="repro"></a>

I've tested this on EC2 `p3.2xlarge` instances (with [an NVIDIA V100 GPU][p3 instances]) using several AMIs:
- [`ami-03fce349214ac583f`]: Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20221226
- [`ami-0a7de320e83dfd4ee`]: Deep Learning AMI GPU PyTorch 1.13.1 (Amazon Linux 2) 20230310
- `ami-003f25e6e2d2db8f1`: NVIDIA GPU-Optimized AMI 22.06.0-676eed8d-dcf5-4784-87d7-0de463205c17 (marketplace image, "subscribe" for free [here](https://aws.amazon.com/marketplace/server/procurement?productId=676eed8d-dcf5-4784-87d7-0de463205c17))
- Several versions of Amazon's ["Deep Learning AMI (Amazon Linux 2)"]:
  - Version 57.1 (`ami-01dfbf223bd1b9835`)
  - Version 61.3 (`ami-0ac44af394b7d6689`)
  - Version 69.1 (`ami-058e8127e717f752b`)

### Create P3-class GPU instance <a id="create-instance"></a>
See instructions below for launching a `p3.2xlarge` instance to reproduce the issue. Other GPU instance types may also exhibit the issue; I've only tested on `p3.2xlarge`s.

#### Using [CDK]
[cdk/] contains [CDK] scripts for launching a `p3.2xlarge` instance
- Uses [`ami-0a7de320e83dfd4ee`] ("Deep Learning AMI GPU PyTorch 1.13.1 (Amazon Linux 2) 20230310") by default
- Runs [`init-conda-env.sh`] on instance boot (make sure you [wait until that's done][cdk#async], on first log in)

#### Using Terraform
- [`instance.tf`] is an example Terraform template for launching a `p3.2xlarge` instance.
- It doesn't initialize the instance, see [instructions below](#setup-instance).

#### Requesting ≥8 vCPU quota for P3-class instances
You may need to request a quota increase for P-class instance vCPUs, if you've never launched one:

```bash
aws service-quotas request-service-quota-increase \
    --service-code ec2 \
    --quota-code L-417A185B \
    --desired-value 8
```

[This page](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-417A185B) should also work.

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
- can take ≈15mins to run

The [cdk/] scripts run the commands above asynchronously during instance boot; [use `tail -f /var/log/cloud-init-output.log` to see when it's done][cdk#async], if you use those scripts.

Here is a snapshot of the Conda env I see, after the above is complete:

<details><summary><code>conda list</code></summary>

```
# packages in environment at /home/ubuntu/miniconda/envs/segfault:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                  2_kmp_llvm    conda-forge
arrow-cpp                 9.0.0           py39hd3ccb9b_2_cpu    conda-forge
aws-c-cal                 0.5.11               h95a6274_0    conda-forge
aws-c-common              0.6.2                h7f98852_0    conda-forge
aws-c-event-stream        0.2.7               h3541f99_13    conda-forge
aws-c-io                  0.10.5               hfb6a706_0    conda-forge
aws-checksums             0.1.11               ha31a3da_7    conda-forge
aws-sdk-cpp               1.8.186              hecaee15_4    conda-forge
blas                      2.116                       mkl    conda-forge
blas-devel                3.9.0            16_linux64_mkl    conda-forge
bokeh                     2.4.3              pyhd8ed1ab_3    conda-forge
brotlipy                  0.7.0           py39hb9d737c_1005    conda-forge
bzip2                     1.0.8                h7f98852_4    conda-forge
c-ares                    1.18.1               h7f98852_0    conda-forge
ca-certificates           2022.12.7            ha878542_0    conda-forge
cachetools                5.3.0              pyhd8ed1ab_0    conda-forge
certifi                   2022.12.7          pyhd8ed1ab_0    conda-forge
cffi                      1.15.1           py39he91dace_3    conda-forge
charset-normalizer        3.1.0              pyhd8ed1ab_0    conda-forge
click                     8.1.3           unix_pyhd8ed1ab_2    conda-forge
cloudpickle               2.2.1              pyhd8ed1ab_0    conda-forge
cryptography              39.0.0           py39hd598818_0    conda-forge
cubinlinker               0.2.2            py39h11215e4_0    rapidsai
cuda                      11.7.1                        0    nvidia
cuda-cccl                 11.7.91                       0    nvidia
cuda-command-line-tools   11.7.1                        0    nvidia
cuda-compiler             11.7.1                        0    nvidia
cuda-cudart               11.7.99                       0    nvidia
cuda-cudart-dev           11.7.99                       0    nvidia
cuda-cuobjdump            11.7.91                       0    nvidia
cuda-cupti                11.7.101                      0    nvidia
cuda-cuxxfilt             11.7.91                       0    nvidia
cuda-demo-suite           12.1.55                       0    nvidia
cuda-documentation        12.1.55                       0    nvidia
cuda-driver-dev           11.7.99                       0    nvidia
cuda-gdb                  12.1.55                       0    nvidia
cuda-libraries            11.7.1                        0    nvidia
cuda-libraries-dev        11.7.1                        0    nvidia
cuda-memcheck             11.8.86                       0    nvidia
cuda-nsight               12.1.55                       0    nvidia
cuda-nsight-compute       12.1.0                        0    nvidia
cuda-nvcc                 11.7.99                       0    nvidia
cuda-nvdisasm             12.1.55                       0    nvidia
cuda-nvml-dev             11.7.91                       0    nvidia
cuda-nvprof               12.1.55                       0    nvidia
cuda-nvprune              11.7.91                       0    nvidia
cuda-nvrtc                11.7.99                       0    nvidia
cuda-nvrtc-dev            11.7.99                       0    nvidia
cuda-nvtx                 11.7.91                       0    nvidia
cuda-nvvp                 12.1.55                       0    nvidia
cuda-python               11.8.1           py39h3fd9d12_0    nvidia
cuda-runtime              11.7.1                        0    nvidia
cuda-sanitizer-api        12.1.55                       0    nvidia
cuda-toolkit              11.7.1                        0    nvidia
cuda-tools                11.7.1                        0    nvidia
cuda-visual-tools         11.7.1                        0    nvidia
cudatoolkit               11.7.0              hd8887f6_10    nvidia
cudf                      22.12.01        cuda_11_py39_gf700408e68_0    rapidsai
cuml                      22.12.00        cuda11_py39_ga9bca9036_0    rapidsai
cupy                      11.6.0           py39hc3c280e_0    conda-forge
cytoolz                   0.12.0           py39hb9d737c_1    conda-forge
dask                      2022.11.1          pyhd8ed1ab_0    conda-forge
dask-core                 2022.11.1          pyhd8ed1ab_0    conda-forge
dask-cuda                 22.12.00        py39_gdc4758e_0    rapidsai
dask-cudf                 22.12.01        cuda_11_py39_gf700408e68_0    rapidsai
distributed               2022.11.1          pyhd8ed1ab_0    conda-forge
dlpack                    0.5                  h9c3ff4c_0    conda-forge
faiss-proc                1.0.0                      cuda    rapidsai
fastavro                  1.7.3            py39h72bdee0_0    conda-forge
fastrlock                 0.8              py39h5a03fae_3    conda-forge
freetype                  2.12.1               hca18f0e_1    conda-forge
fsspec                    2023.4.0           pyh1a96a4e_0    conda-forge
gds-tools                 1.6.0.25                      0    nvidia
gflags                    2.2.2             he1b5a44_1004    conda-forge
glog                      0.6.0                h6f12383_0    conda-forge
grpc-cpp                  1.47.1               hbad87ad_6    conda-forge
heapdict                  1.0.1                      py_0    conda-forge
icu                       72.1                 hcb278e6_0    conda-forge
idna                      3.4                pyhd8ed1ab_0    conda-forge
jinja2                    3.1.2              pyhd8ed1ab_1    conda-forge
joblib                    1.2.0              pyhd8ed1ab_0    conda-forge
keyutils                  1.6.1                h166bdaf_0    conda-forge
krb5                      1.20.1               hf9c8cef_0    conda-forge
lcms2                     2.15                 haa2dc70_1    conda-forge
ld_impl_linux-64          2.40                 h41732ed_0    conda-forge
lerc                      4.0.0                h27087fc_0    conda-forge
libabseil                 20220623.0      cxx17_h05df665_6    conda-forge
libblas                   3.9.0            16_linux64_mkl    conda-forge
libbrotlicommon           1.0.9                h166bdaf_8    conda-forge
libbrotlidec              1.0.9                h166bdaf_8    conda-forge
libbrotlienc              1.0.9                h166bdaf_8    conda-forge
libcblas                  3.9.0            16_linux64_mkl    conda-forge
libcrc32c                 1.1.2                h9c3ff4c_0    conda-forge
libcublas                 12.1.0.26                     0    nvidia
libcublas-dev             12.1.0.26                     0    nvidia
libcudf                   22.12.01        cuda11_gf700408e68_0    rapidsai
libcufft                  11.0.2.4                      0    nvidia
libcufft-dev              11.0.2.4                      0    nvidia
libcufile                 1.6.0.25                      0    nvidia
libcufile-dev             1.6.0.25                      0    nvidia
libcuml                   22.12.00        cuda11_ga9bca9036_0    rapidsai
libcumlprims              22.12.00        cuda11_gf010d79_0    nvidia
libcurand                 10.3.2.56                     0    nvidia
libcurand-dev             10.3.2.56                     0    nvidia
libcurl                   7.87.0               h6312ad2_0    conda-forge
libcusolver               11.4.1.48                     0    nvidia
libcusolver-dev           11.4.1.48                     0    nvidia
libcusparse               11.7.5.86                     0    nvidia
libcusparse-dev           11.7.5.86                     0    nvidia
libdeflate                1.18                 h0b41bf4_0    conda-forge
libedit                   3.1.20191231         he28a2e2_2    conda-forge
libev                     4.33                 h516909a_1    conda-forge
libevent                  2.1.10               h9b69904_4    conda-forge
libfaiss                  1.7.0           cuda112h5bea7ad_8_cuda    conda-forge
libffi                    3.4.2                h7f98852_5    conda-forge
libgcc-ng                 12.2.0              h65d4601_19    conda-forge
libgfortran-ng            12.2.0              h69a702a_19    conda-forge
libgfortran5              12.2.0              h337968e_19    conda-forge
libgoogle-cloud           2.1.0                h9ebe8e8_2    conda-forge
libhwloc                  2.9.0                hd6dc26d_0    conda-forge
libiconv                  1.17                 h166bdaf_0    conda-forge
libjpeg-turbo             2.1.5.1              h0b41bf4_0    conda-forge
liblapack                 3.9.0            16_linux64_mkl    conda-forge
liblapacke                3.9.0            16_linux64_mkl    conda-forge
libllvm11                 11.1.0               he0ac6c6_5    conda-forge
libnghttp2                1.51.0               hdcd2b5c_0    conda-forge
libnpp                    12.0.2.50                     0    nvidia
libnpp-dev                12.0.2.50                     0    nvidia
libnsl                    2.0.0                h7f98852_0    conda-forge
libnvjpeg                 12.1.0.39                     0    nvidia
libnvjpeg-dev             12.1.0.39                     0    nvidia
libpng                    1.6.39               h753d276_0    conda-forge
libprotobuf               3.20.2               h6239696_0    conda-forge
libraft-distance          22.12.01        cuda11_ga655c9a7_0    rapidsai
libraft-headers           22.12.01        cuda11_ga655c9a7_0    rapidsai
libraft-nn                22.12.01        cuda11_ga655c9a7_0    rapidsai
librmm                    22.12.00        cuda11_g8aae42d1_0    rapidsai
libsqlite                 3.40.0               h753d276_0    conda-forge
libssh2                   1.10.0               haa6b8db_3    conda-forge
libstdcxx-ng              12.2.0              h46fd767_19    conda-forge
libthrift                 0.16.0               h491838f_2    conda-forge
libtiff                   4.5.0                ha587672_6    conda-forge
libutf8proc               2.8.0                h166bdaf_0    conda-forge
libuuid                   2.38.1               h0b41bf4_0    conda-forge
libwebp-base              1.3.0                h0b41bf4_0    conda-forge
libxcb                    1.13              h7f98852_1004    conda-forge
libxml2                   2.10.3               hfdac1af_6    conda-forge
libzlib                   1.2.13               h166bdaf_4    conda-forge
llvm-openmp               16.0.1               h417c0b6_0    conda-forge
llvmlite                  0.39.1           py39h7d9a04d_1    conda-forge
locket                    1.0.0              pyhd8ed1ab_0    conda-forge
lz4                       4.3.2            py39h724f13c_0    conda-forge
lz4-c                     1.9.4                hcb278e6_0    conda-forge
markupsafe                2.1.2            py39h72bdee0_0    conda-forge
mkl                       2022.1.0           h84fe81f_915    conda-forge
mkl-devel                 2022.1.0           ha770c72_916    conda-forge
mkl-include               2022.1.0           h84fe81f_915    conda-forge
msgpack-python            1.0.5            py39h4b4f3f3_0    conda-forge
nccl                      2.14.3.1             h0800d71_0    conda-forge
ncurses                   6.3                  h27087fc_1    conda-forge
nsight-compute            2023.1.0.15                   0    nvidia
numba                     0.56.4           py39h71a7301_1    conda-forge
numpy                     1.23.5           py39h3d75532_0    conda-forge
nvtx                      0.2.3            py39hb9d737c_2    conda-forge
openjpeg                  2.5.0                hfec8fc6_2    conda-forge
openssl                   1.1.1t               h0b41bf4_0    conda-forge
orc                       1.7.6                h6c59b99_0    conda-forge
packaging                 23.0               pyhd8ed1ab_0    conda-forge
pandas                    1.5.3            py39h2ad29b5_1    conda-forge
parquet-cpp               1.5.1                         2    conda-forge
partd                     1.3.0              pyhd8ed1ab_0    conda-forge
pillow                    9.5.0            py39h7207d5c_0    conda-forge
pip                       23.0.1             pyhd8ed1ab_0    conda-forge
platformdirs              3.2.0              pyhd8ed1ab_0    conda-forge
pooch                     1.7.0              pyha770c72_3    conda-forge
protobuf                  3.20.2           py39h5a03fae_1    conda-forge
psutil                    5.9.4            py39hb9d737c_0    conda-forge
pthread-stubs             0.4               h36c2ea0_1001    conda-forge
ptxcompiler               0.7.0            py39h2405124_3    conda-forge
pyarrow                   9.0.0           py39hc0775d8_2_cpu    conda-forge
pycparser                 2.21               pyhd8ed1ab_0    conda-forge
pylibraft                 22.12.01        cuda11_py39_ga655c9a7_0    rapidsai
pynvml                    11.5.0             pyhd8ed1ab_0    conda-forge
pyopenssl                 23.1.1             pyhd8ed1ab_0    conda-forge
pysocks                   1.7.1              pyha2e5f31_6    conda-forge
python                    3.9.13          h9a8a25e_0_cpython    conda-forge
python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
python_abi                3.9                      3_cp39    conda-forge
pytorch                   1.13.1          py3.9_cuda11.7_cudnn8.5.0_0    pytorch
pytorch-cuda              11.7                 h67b0de4_0    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2023.3             pyhd8ed1ab_0    conda-forge
pyyaml                    6.0              py39hb9d737c_5    conda-forge
raft-dask                 22.12.01        cuda11_py39_ga655c9a7_0    rapidsai
re2                       2022.06.01           h27087fc_1    conda-forge
readline                  8.2                  h8228510_1    conda-forge
requests                  2.28.2             pyhd8ed1ab_1    conda-forge
rmm                       22.12.00        cuda11_py39_g8aae42d1_0    rapidsai
s2n                       1.0.10               h9b69904_0    conda-forge
scipy                     1.10.1           py39h7360e5f_0    conda-forge
setuptools                67.6.1             pyhd8ed1ab_0    conda-forge
six                       1.16.0             pyh6c4a22f_0    conda-forge
snappy                    1.1.10               h9fff704_0    conda-forge
sortedcontainers          2.4.0              pyhd8ed1ab_0    conda-forge
spdlog                    1.8.5                h4bd325d_1    conda-forge
sqlite                    3.40.0               h4ff8645_0    conda-forge
tbb                       2021.8.0             hf52228f_0    conda-forge
tblib                     1.7.0              pyhd8ed1ab_0    conda-forge
tk                        8.6.12               h27826a3_0    conda-forge
toolz                     0.12.0             pyhd8ed1ab_0    conda-forge
tornado                   6.1              py39hb9d737c_3    conda-forge
treelite                  3.0.1            py39hc7ff369_0    conda-forge
treelite-runtime          3.0.0                    pypi_0    pypi
typing-extensions         4.5.0                hd8ed1ab_0    conda-forge
typing_extensions         4.5.0              pyha770c72_0    conda-forge
tzdata                    2023c                h71feb2d_0    conda-forge
ucx                       1.13.1               h538f049_1    conda-forge
ucx-proc                  1.0.0                       gpu    rapidsai
ucx-py                    0.29.00         py39_g01ac0ef_0    rapidsai
urllib3                   1.26.15            pyhd8ed1ab_0    conda-forge
wheel                     0.40.0             pyhd8ed1ab_0    conda-forge
xorg-libxau               1.0.9                h7f98852_0    conda-forge
xorg-libxdmcp             1.1.3                h7f98852_0    conda-forge
xz                        5.2.6                h166bdaf_0    conda-forge
yaml                      0.2.5                h7f98852_2    conda-forge
zict                      2.2.0              pyhd8ed1ab_0    conda-forge
zlib                      1.2.13               h166bdaf_4    conda-forge
zstd                      1.5.2                h3eb15da_6    conda-forge
```
</details>

<details><summary><code>conda list --export</code></summary>

```
conda list --export
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
_libgcc_mutex=0.1=conda_forge
_openmp_mutex=4.5=2_kmp_llvm
arrow-cpp=9.0.0=py39hd3ccb9b_2_cpu
aws-c-cal=0.5.11=h95a6274_0
aws-c-common=0.6.2=h7f98852_0
aws-c-event-stream=0.2.7=h3541f99_13
aws-c-io=0.10.5=hfb6a706_0
aws-checksums=0.1.11=ha31a3da_7
aws-sdk-cpp=1.8.186=hecaee15_4
blas=2.116=mkl
blas-devel=3.9.0=16_linux64_mkl
bokeh=2.4.3=pyhd8ed1ab_3
brotlipy=0.7.0=py39hb9d737c_1005
bzip2=1.0.8=h7f98852_4
c-ares=1.18.1=h7f98852_0
ca-certificates=2022.12.7=ha878542_0
cachetools=5.3.0=pyhd8ed1ab_0
certifi=2022.12.7=pyhd8ed1ab_0
cffi=1.15.1=py39he91dace_3
charset-normalizer=3.1.0=pyhd8ed1ab_0
click=8.1.3=unix_pyhd8ed1ab_2
cloudpickle=2.2.1=pyhd8ed1ab_0
cryptography=39.0.0=py39hd598818_0
cubinlinker=0.2.2=py39h11215e4_0
cuda=11.7.1=0
cuda-cccl=11.7.91=0
cuda-command-line-tools=11.7.1=0
cuda-compiler=11.7.1=0
cuda-cudart=11.7.99=0
cuda-cudart-dev=11.7.99=0
cuda-cuobjdump=11.7.91=0
cuda-cupti=11.7.101=0
cuda-cuxxfilt=11.7.91=0
cuda-demo-suite=12.1.55=0
cuda-documentation=12.1.55=0
cuda-driver-dev=11.7.99=0
cuda-gdb=12.1.55=0
cuda-libraries=11.7.1=0
cuda-libraries-dev=11.7.1=0
cuda-memcheck=11.8.86=0
cuda-nsight=12.1.55=0
cuda-nsight-compute=12.1.0=0
cuda-nvcc=11.7.99=0
cuda-nvdisasm=12.1.55=0
cuda-nvml-dev=11.7.91=0
cuda-nvprof=12.1.55=0
cuda-nvprune=11.7.91=0
cuda-nvrtc=11.7.99=0
cuda-nvrtc-dev=11.7.99=0
cuda-nvtx=11.7.91=0
cuda-nvvp=12.1.55=0
cuda-python=11.8.1=py39h3fd9d12_0
cuda-runtime=11.7.1=0
cuda-sanitizer-api=12.1.55=0
cuda-toolkit=11.7.1=0
cuda-tools=11.7.1=0
cuda-visual-tools=11.7.1=0
cudatoolkit=11.7.0=hd8887f6_10
cudf=22.12.01=cuda_11_py39_gf700408e68_0
cuml=22.12.00=cuda11_py39_ga9bca9036_0
cupy=11.6.0=py39hc3c280e_0
cytoolz=0.12.0=py39hb9d737c_1
dask=2022.11.1=pyhd8ed1ab_0
dask-core=2022.11.1=pyhd8ed1ab_0
dask-cuda=22.12.00=py39_gdc4758e_0
dask-cudf=22.12.01=cuda_11_py39_gf700408e68_0
distributed=2022.11.1=pyhd8ed1ab_0
dlpack=0.5=h9c3ff4c_0
faiss-proc=1.0.0=cuda
fastavro=1.7.3=py39h72bdee0_0
fastrlock=0.8=py39h5a03fae_3
freetype=2.12.1=hca18f0e_1
fsspec=2023.4.0=pyh1a96a4e_0
gds-tools=1.6.0.25=0
gflags=2.2.2=he1b5a44_1004
glog=0.6.0=h6f12383_0
grpc-cpp=1.47.1=hbad87ad_6
heapdict=1.0.1=py_0
icu=72.1=hcb278e6_0
idna=3.4=pyhd8ed1ab_0
jinja2=3.1.2=pyhd8ed1ab_1
joblib=1.2.0=pyhd8ed1ab_0
keyutils=1.6.1=h166bdaf_0
krb5=1.20.1=hf9c8cef_0
lcms2=2.15=haa2dc70_1
ld_impl_linux-64=2.40=h41732ed_0
lerc=4.0.0=h27087fc_0
libabseil=20220623.0=cxx17_h05df665_6
libblas=3.9.0=16_linux64_mkl
libbrotlicommon=1.0.9=h166bdaf_8
libbrotlidec=1.0.9=h166bdaf_8
libbrotlienc=1.0.9=h166bdaf_8
libcblas=3.9.0=16_linux64_mkl
libcrc32c=1.1.2=h9c3ff4c_0
libcublas=12.1.0.26=0
libcublas-dev=12.1.0.26=0
libcudf=22.12.01=cuda11_gf700408e68_0
libcufft=11.0.2.4=0
libcufft-dev=11.0.2.4=0
libcufile=1.6.0.25=0
libcufile-dev=1.6.0.25=0
libcuml=22.12.00=cuda11_ga9bca9036_0
libcumlprims=22.12.00=cuda11_gf010d79_0
libcurand=10.3.2.56=0
libcurand-dev=10.3.2.56=0
libcurl=7.87.0=h6312ad2_0
libcusolver=11.4.1.48=0
libcusolver-dev=11.4.1.48=0
libcusparse=11.7.5.86=0
libcusparse-dev=11.7.5.86=0
libdeflate=1.18=h0b41bf4_0
libedit=3.1.20191231=he28a2e2_2
libev=4.33=h516909a_1
libevent=2.1.10=h9b69904_4
libfaiss=1.7.0=cuda112h5bea7ad_8_cuda
libffi=3.4.2=h7f98852_5
libgcc-ng=12.2.0=h65d4601_19
libgfortran-ng=12.2.0=h69a702a_19
libgfortran5=12.2.0=h337968e_19
libgoogle-cloud=2.1.0=h9ebe8e8_2
libhwloc=2.9.0=hd6dc26d_0
libiconv=1.17=h166bdaf_0
libjpeg-turbo=2.1.5.1=h0b41bf4_0
liblapack=3.9.0=16_linux64_mkl
liblapacke=3.9.0=16_linux64_mkl
libllvm11=11.1.0=he0ac6c6_5
libnghttp2=1.51.0=hdcd2b5c_0
libnpp=12.0.2.50=0
libnpp-dev=12.0.2.50=0
libnsl=2.0.0=h7f98852_0
libnvjpeg=12.1.0.39=0
libnvjpeg-dev=12.1.0.39=0
libpng=1.6.39=h753d276_0
libprotobuf=3.20.2=h6239696_0
libraft-distance=22.12.01=cuda11_ga655c9a7_0
libraft-headers=22.12.01=cuda11_ga655c9a7_0
libraft-nn=22.12.01=cuda11_ga655c9a7_0
librmm=22.12.00=cuda11_g8aae42d1_0
libsqlite=3.40.0=h753d276_0
libssh2=1.10.0=haa6b8db_3
libstdcxx-ng=12.2.0=h46fd767_19
libthrift=0.16.0=h491838f_2
libtiff=4.5.0=ha587672_6
libutf8proc=2.8.0=h166bdaf_0
libuuid=2.38.1=h0b41bf4_0
libwebp-base=1.3.0=h0b41bf4_0
libxcb=1.13=h7f98852_1004
libxml2=2.10.3=hfdac1af_6
libzlib=1.2.13=h166bdaf_4
llvm-openmp=16.0.1=h417c0b6_0
llvmlite=0.39.1=py39h7d9a04d_1
locket=1.0.0=pyhd8ed1ab_0
lz4=4.3.2=py39h724f13c_0
lz4-c=1.9.4=hcb278e6_0
markupsafe=2.1.2=py39h72bdee0_0
mkl=2022.1.0=h84fe81f_915
mkl-devel=2022.1.0=ha770c72_916
mkl-include=2022.1.0=h84fe81f_915
msgpack-python=1.0.5=py39h4b4f3f3_0
nccl=2.14.3.1=h0800d71_0
ncurses=6.3=h27087fc_1
nsight-compute=2023.1.0.15=0
numba=0.56.4=py39h71a7301_1
numpy=1.23.5=py39h3d75532_0
nvtx=0.2.3=py39hb9d737c_2
openjpeg=2.5.0=hfec8fc6_2
openssl=1.1.1t=h0b41bf4_0
orc=1.7.6=h6c59b99_0
packaging=23.0=pyhd8ed1ab_0
pandas=1.5.3=py39h2ad29b5_1
parquet-cpp=1.5.1=2
partd=1.3.0=pyhd8ed1ab_0
pillow=9.5.0=py39h7207d5c_0
pip=23.0.1=pyhd8ed1ab_0
platformdirs=3.2.0=pyhd8ed1ab_0
pooch=1.7.0=pyha770c72_3
protobuf=3.20.2=py39h5a03fae_1
psutil=5.9.4=py39hb9d737c_0
pthread-stubs=0.4=h36c2ea0_1001
ptxcompiler=0.7.0=py39h2405124_3
pyarrow=9.0.0=py39hc0775d8_2_cpu
pycparser=2.21=pyhd8ed1ab_0
pylibraft=22.12.01=cuda11_py39_ga655c9a7_0
pynvml=11.5.0=pyhd8ed1ab_0
pyopenssl=23.1.1=pyhd8ed1ab_0
pysocks=1.7.1=pyha2e5f31_6
python=3.9.13=h9a8a25e_0_cpython
python-dateutil=2.8.2=pyhd8ed1ab_0
python_abi=3.9=3_cp39
pytorch=1.13.1=py3.9_cuda11.7_cudnn8.5.0_0
pytorch-cuda=11.7=h67b0de4_0
pytorch-mutex=1.0=cuda
pytz=2023.3=pyhd8ed1ab_0
pyyaml=6.0=py39hb9d737c_5
raft-dask=22.12.01=cuda11_py39_ga655c9a7_0
re2=2022.06.01=h27087fc_1
readline=8.2=h8228510_1
requests=2.28.2=pyhd8ed1ab_1
rmm=22.12.00=cuda11_py39_g8aae42d1_0
s2n=1.0.10=h9b69904_0
scipy=1.10.1=py39h7360e5f_0
setuptools=67.6.1=pyhd8ed1ab_0
six=1.16.0=pyh6c4a22f_0
snappy=1.1.10=h9fff704_0
sortedcontainers=2.4.0=pyhd8ed1ab_0
spdlog=1.8.5=h4bd325d_1
sqlite=3.40.0=h4ff8645_0
tbb=2021.8.0=hf52228f_0
tblib=1.7.0=pyhd8ed1ab_0
tk=8.6.12=h27826a3_0
toolz=0.12.0=pyhd8ed1ab_0
tornado=6.1=py39hb9d737c_3
treelite=3.0.1=py39hc7ff369_0
treelite-runtime=3.0.0=pypi_0
typing-extensions=4.5.0=hd8ed1ab_0
typing_extensions=4.5.0=pyha770c72_0
tzdata=2023c=h71feb2d_0
ucx=1.13.1=h538f049_1
ucx-proc=1.0.0=gpu
ucx-py=0.29.00=py39_g01ac0ef_0
urllib3=1.26.15=pyhd8ed1ab_0
wheel=0.40.0=pyhd8ed1ab_0
xorg-libxau=1.0.9=h7f98852_0
xorg-libxdmcp=1.1.3=h7f98852_0
xz=5.2.6=h166bdaf_0
yaml=0.2.5=h7f98852_2
zict=2.2.0=pyhd8ed1ab_0
zlib=1.2.13=h166bdaf_4
zstd=1.5.2=h3eb15da_6
```
</details>

### Reproduce segfault on host <a id="host"></a>
[`run.py`] runs [`neighbors.py`] repeatedly, exhibiting occasional (≈10%) segfaults:
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

I've also built+pushed this image as [runsascoded/torch-cuml-gpu-segfault](https://hub.docker.com/r/runsascoded/torch-cuml-gpu-segfault/tags).

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

This can be verified via:
```bash
./run.py -T  # skip the unused `import torch`; segfault no longer occurs
```

If [the `nn.kneighbors` call](neighbors.py#L23) is commented out, the segfault also goes away. Perhaps there is a "use after free" problem with some data structures instantiated by `cuml` (probably `cudf` DataFrames)?

### Minimizing the example <a id="minimizing"></a>
This is as minimal of a repro as I've found for this issue (which initially manifested in a [Metaflow] pipeline, during a call to [`scanpy.preprocessing.neighbors`], on larger, proprietary data).

I've tried to reduce dependencies and update versions of what remains (see [`environment.yml`]). I originally hit the issue with CUDA 11.6, `cuml==22.06.01`, and Torch 1.12.1, but the repro persists on CUDA 11.7, `cuml==22.12.00`, and Torch 1.13.1.

### Python `faulthandler` not working <a id="faulthandler"></a>
I've tried to enable a more detailed stack trace from the segfault in a few places ([Dockerfile#L4](Dockerfile#L13), [neighbors.py#L7](neighbors.py#L6)), per [these instructions][segfault debug article], but have so far been unable to get any more info about where the segfault is occurring.

### Side-effectful `import torch` breaks with RAPIDS <a id="torch-vs-rapids"></a>
It would be good to understand what side effects `import torch` has, which causes this subtle issue with subsequent cuml/RAPIDS execution (even when the imported `torch` is never used).

In 2022 I ran into a similar issue (I believe with Torch 1.10 or 1.11, CUDA 11.4, and RAPIDS 21.x) where `import cuml; import torch` was fine, but `import torch; import cuml` raised a linker error. It seems like there are some deep and brittle interactions between these libraries.

### Sensitivity calculations <a id="sensitivity"></a>
I settled on exercising `neighbors.py` 30x to determine whether a given repro candidate exhibits the segfault. My 10% estimate (of the how likely a given run is to segfault) has remained fairly accurate and stable as I reduced the repro by several orders of magnitude (in terms of lines of code, and overall complexity).

The chance of a false positive (code falsely appears to work / not segfault) is 0.9^30 ≈ 4%.
- In practice I believe I've run this 30x trial 100 times and only seen 1-2 such FPs (30 runs, no failures).
- I ran it 400x once, and observed 42 failures with a maximum of 47 consecutive successes between two segfaults.

Bumping the number of repetitions to e.g. 50 (`./run.py -n50`) can further reduce uncertainty; in the common case, the first segfault appears much earlier, allowing for short-circuiting or initiating a subsequent run.


[`scanpy.preprocessing.neighbors`]: https://github.com/scverse/scanpy/blob/1.8.2/scanpy/neighbors/__init__.py#L52
[`scanpy.neighbors.compute_neighbors_rapids`]: https://github.com/scverse/scanpy/blob/1.8.2/scanpy/neighbors/__init__.py#L318
[`environment.yml`]: environment.yml
[`cuml.neighbors.NearestNeighbors`]: https://github.com/rapidsai/cuml/blob/v22.06.01/python/cuml/neighbors/nearest_neighbors.pyx#L153
[`kneighbors`]: https://github.com/rapidsai/cuml/blob/v22.06.01/python/cuml/neighbors/nearest_neighbors.pyx#L482
[`import torch`]: neighbors.py#L15
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
[PyTorch]: https://pytorch.org/
[`cuml`]: https://github.com/rapidsai/cuml
