#!/usr/bin/env bash
# Given a stack name, look up its managed instance and key-pair, and SSH into the instance:
#
# . cdk-ssh.sh [stack name]

stack="${1:-GpuSegfaultStack}"
echo "stack: $stack"

# Key name from Stack
key_name="$(aws cloudformation describe-stack-resources --stack-name $stack | jq -r '.StackResources[]|select(.ResourceType=="AWS::EC2::KeyPair").PhysicalResourceId')"
echo "key_name: $key_name"

key_pair_id="$(aws ec2 describe-key-pairs | jq -r ".KeyPairs[]|select(.KeyName==\"$key_name\").KeyPairId")"
echo "key_pair_id: $key_pair_id"

key_file=gpu-segfault-key-pair
aws ssm get-parameter --name /ec2/keypair/$key_pair_id --with-decryption --output text --query Parameter.Value > $key_file
chmod 600 $key_file

instance_id="$(aws cloudformation describe-stack-resources --stack-name $stack | jq -r '.StackResources[]|select(.ResourceType=="AWS::EC2::Instance").PhysicalResourceId')"
echo "instance_id: $instance_id"

public_ip="$(aws ec2 describe-instances --instance-ids $instance_id | jq -r '.Reservations[].Instances[0].PublicIpAddress')"
echo "public_ip: $public_ip"

cmd=(ssh -oForwardAgent=yes -oIdentitiesOnly=yes -oStrictHostKeyChecking=no "-i$key_file" "ubuntu@$public_ip")
echo "${cmd[*]}"
"${cmd[@]}"
