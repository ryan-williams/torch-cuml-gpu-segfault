# torch/cuml segfault CDK setup
- [Usage](#usage)
    - [Deploy](#deploy)
    - [SSH into launched instance](#ssh)
        - [Note that the setup script runs asynchronously!](#async)
    - [Run test script](#test)
    - [Destroy](#destroy)
- [Customizing the deployed stack](#customize)

## Usage <a id="usage"></a>

### Deploy <a id="deploy"></a>
```bash
cdk deploy  # deploy GpuSegfaultStack
```

This boots a `p3.2xlarge` instance (containing an NVIDIA V100 GPU), and runs a setup script.

### SSH into launched instance <a id="ssh"></a>
[`cdk-ssh.sh`](cdk-ssh.sh) is an easy way to SSH into the launched instance:
```bash
. cdk-ssh.sh [stack name]
```

#### Note that the setup script runs asynchronously! <a id="async"></a>
[The instance setup script](gpu_segfault/gpu_segfault_stack.py#L59-L71) clones this repo and runs [init-conda-env.sh](../init-conda-env.sh) at launch, which can take ≈15 minutes.

When you first log SSH in, it's a good idea to:
```bash
tail -f /var/log/cloud-init-output.log
```
and wait until you see a line like this, declaring that cloud-init has finished:
```
Cloud-init v. 22.4.2-0ubuntu0~20.04.2 finished at Sat, 08 Apr 2023 20:51:52 +0000
```

Once cloud-init is done, a fresh login (or `. ~/.bashrc`) should activate the `segfault` conda env, with necessary dependencies installed (see [`environment.yml`](../environment.yml)).

### Run test script <a id="test"></a>
```bash
./run.py -q  # exercise `neighbors.py` 30x, observe segfaults
```
See [../README.md](../README.md#host) for more info.

### Destroy <a id="destroy"></a>
```bash
cdk destroy # destroy GpuSegfaultStack
```

## Customizing the deployed stack <a id="customize"></a>
[`app.py`](app.py) supports various CDK context vars for customizing the stack, e.g. to test with [an Amazon Linux Torch 1.13.1 AMI](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-1-13-amazon-linux-2/) (instead of [the default Ubuntu Torch image](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-1-13-ubuntu-20-04/)):

```bash
cdk deploy \
    -c stack=azlinux \
    -c ami_id=ami-0a7de320e83dfd4ee \
    -c user=ec2-user
```
