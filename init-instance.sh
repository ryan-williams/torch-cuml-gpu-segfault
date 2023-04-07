#!/usr/bin/env bash

set -ex

# Clone this repo
git clone https://github.com/ryan-williams/torch-cuml-metaflow-gpu-segfault gpu-segfault
cd gpu-segfault
./init-conda-env.sh
