#!/usr/bin/env python3
import json
from os import environ as env
from subprocess import check_output

from aws_cdk import App

from gpu_segfault.gpu_segfault_stack import GpuSegfaultStack

app = App()
def get_context(key, default=None):
    return app.node.try_get_context(key) or default


# AMIs:
# - ami-03fce349214ac583f: Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20221226
# - ami-0a7de320e83dfd4ee: Deep Learning AMI GPU PyTorch 1.13.1 (Amazon Linux 2) 20230310
# - ami-003f25e6e2d2db8f1: NVIDIA GPU-Optimized AMI 22.06.0-676eed8d-dcf5-4784-87d7-0de463205c17 ("subscribe" for free: https://aws.amazon.com/marketplace/server/procurement?productId=676eed8d-dcf5-4784-87d7-0de463205c17)
# - ami-007855ac798b5175e: ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-20230325 (simple Ubuntu, non-GPU, for CDK testing)
ami_id = get_context('ami_id', 'ami-03fce349214ac583f')  # Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20221226

ami = json.loads(check_output(['aws', 'ec2', 'describe-images', '--image-ids', ami_id]).decode())['Images'][0]
ami_name = ami['Name']
print(f'{ami_name=}')

stack = get_context('stack', "GpuSegfaultStack")
instance_type = get_context('instance_type', 'p3.2xlarge')
user = get_context('user', 'ubuntu')
clone_ref = get_context('clone_ref')
environment_file = get_context('environment_file')
block_device_mapping = ami['BlockDeviceMappings'][0]
volume_size = get_context('volume_size', block_device_mapping['Ebs']['VolumeSize'] + 40)
device_name = get_context('device_name', block_device_mapping['DeviceName'])

account = env.get('CDK_DEFAULT_ACCOUNT')
if not account:
    account = env.get('AWS_PROFILE')
    if not account:
        account = json.loads(check_output(['aws', 'sts', 'get-caller-identity']).decode())['Account']
        if not account:
            raise RuntimeError("No $CDK_DEFAULT_ACCOUNT or $AWS_PROFILE found")
region = env.get('CDK_DEFAULT_REGION', 'us-east-1')

env = { 'account': account, 'region': region, }
print(f'context vars: {ami_id=}, {volume_size=}, {device_name=}, {user=}. {env=}')
GpuSegfaultStack(
    app,
    stack,
    env=env,
    ami_id=ami_id,
    instance_type=instance_type,
    volume_size=volume_size,
    device_name=device_name,
    user=user,
    clone_ref=clone_ref,
    environment_file=environment_file,
)
app.synth()
