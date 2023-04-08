#!/usr/bin/env bash
# Destroy a test stack created by deploy-test.sh:
#
# $ destroy-test.sh [stack name]
#
# Essentially runs:
#
# $ cdk destroy -c stack=$stack -c ami_id=ami-007855ac798b5175e -c instance_type=t2.2xlarge -c user=ubuntu -c clone_ref=test -o $stack.cdk $stack

if [ $# -eq 0 ]; then
    stack=
    args=()
else
    stack="${@:(($#))}"
    args=("${@:1:$(($#-1))}" --force -c stack=$stack -c environment_file=environment-test.yml)
fi

args+=(-c ami_id=ami-007855ac798b5175e -c instance_type=t2.2xlarge -c user=ubuntu -c clone_ref=test)
if [ -n "$stack" ]; then
    args+=(-o $stack.cdk "$stack")
fi

cmd=(cdk destroy "${args[@]}")
echo "Running: ${cmd[*]}"
"${cmd[@]}"
