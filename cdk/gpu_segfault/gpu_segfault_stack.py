from aws_cdk import Stack
from aws_cdk.aws_ec2 import MachineImage, Vpc, Peer, SecurityGroup, Instance, InstanceType, Port, CfnKeyPair, \
    BlockDevice, BlockDeviceVolume, EbsDeviceProps, EbsDeviceVolumeType, UserData
from constructs import Construct


class GpuSegfaultStack(Stack):
    def __init__(
            self,
            scope: Construct,
            id: str,
            ami_id,
            volume_size,
            sec_grp=None,
            vpc_id=None,
            instance_name=None,
            instance_type='p3.2xlarge',
            key_name=None,
            device_name='/dev/xvda',
            user='ec2-user',
            clone_ref=None,
            environment_file=None,
            **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        sec_grp = sec_grp or f'{id}-sg'
        instance_name = instance_name or f'{id}-instance'
        key_name = key_name or f'{id}-key-pair'

        home = f'/home/{user}'
        CfnKeyPair(self, key_name, key_name=key_name)

        ami_image = MachineImage().lookup(name='*', filters={'image-id': [ami_id]})
        if not ami_image:
            raise RuntimeError('Failed finding AMI image')

        instance_type = InstanceType(instance_type)
        if not instance_type:
            raise RuntimeError('Failed finding instance')

        vpc_kwargs = dict(vpc_id=vpc_id) if vpc_id else dict(is_default=True)
        vpc = Vpc.from_lookup(self, 'vpc', **vpc_kwargs)
        if not vpc:
            raise RuntimeError(f'Failed finding VPC {vpc_id}')

        sec_grp = SecurityGroup(self, sec_grp, vpc=vpc, allow_all_outbound=True)
        if not sec_grp:
            raise RuntimeError(f'Failed finding security group {sec_grp}')

        sec_grp.add_ingress_rule(
            peer=Peer.ipv4('0.0.0.0/0'),
            description='inbound SSH',
            connection=Port.tcp(22),
        )
        if not sec_grp:
            raise RuntimeError('Failed creating security group')

        user_data = UserData.for_linux()
        user_data.add_commands(
            f'export "HOME={home}"',
            f'cd "{home}"',
            'echo "user_data: cloning (HOME: $HOME, PWD: $PWD)"',
            f'git clone {f"-b {clone_ref} " if clone_ref else ""}https://github.com/ryan-williams/torch-cuml-metaflow-gpu-segfault gpu-segfault',
            'cd gpu-segfault',
            f'chown -R {user}:{user} .',
            'echo "user_data: initializing conda"',
            f'./init-conda-env.sh{f" {environment_file}" if environment_file else ""}',
            f'chown -R {user}:{user} ~/miniconda ~/.conda',
            'echo "user_data: done"',
        )

        ec2_inst = Instance(
            self, instance_name,
            instance_name=instance_name,
            instance_type=instance_type,
            machine_image=ami_image,
            vpc=vpc,
            security_group=sec_grp,
            key_name=key_name,
            user_data=user_data,
            block_devices=[BlockDevice(
                device_name=device_name,
                volume=BlockDeviceVolume(
                    ebs_device=EbsDeviceProps(
                        volume_type=EbsDeviceVolumeType.GP3,
                        volume_size=volume_size,
                    ),
                ),
            )],
        )
        if not ec2_inst:
            raise RuntimeError('Failed creating ec2 instance')
