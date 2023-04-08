#!/usr/bin/env bash
# Deploy a test stack (t2.2xlarge instance, no GPU), for e.g. debugging instance "user data" startup script:
#
# $ deploy-test.sh [stack name]
#
# Essentially runs:
#
# $ cdk deploy -c stack=$stack -c ami_id=ami-007855ac798b5175e -c instance_type=t2.2xlarge -c user=ubuntu -c clone_ref=test -o $stack.cdk

if [ $# -eq 0 ]; then
    stack=
    args=()
else
    stack="${@:(($#))}"
    args=("${@:1:$(($#-1))}" --require-approval never -c stack=$stack -c environment_file=environment-test.yml)
fi

args+=(-c ami_id=ami-007855ac798b5175e -c instance_type=t2.2xlarge -c user=ubuntu -c clone_ref=test)
if [ -n "$stack" ]; then
    args+=(-o $stack.cdk)
fi

cmd=(cdk deploy "${args[@]}")
echo "Running: ${cmd[*]}"
"${cmd[@]}"
