#!/usr/bin/env bash

set -ex

# Install recent Conda + libmamba-solver
d=~/miniconda
wget -q "https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh" -O ~/miniconda.sh
/bin/bash ~/miniconda.sh -b -p $d
rm ~/miniconda.sh
echo ". $d/etc/profile.d/conda.sh" >> ~/.bashrc
. ~/.bashrc
conda="$d/bin/conda"
time "$conda" install -y -n base conda-libmamba-solver
"$conda" config --set solver libmamba
"$conda" config --set channel_priority flexible  # https://github.com/rapidsai/cuml/issues/4016

if [ $# -eq 1 ]; then
    args=(-f "$1")
elif [ $# -eq 0 ]; then
    args=()
else
    echo "Usage: $0 [environment-file]" >&2
    exit 1
fi
# Create conda env with necessary dependencies (see environment.yml)
time "$conda" env update -n segfault "${args[@]}"
echo "conda activate segfault" >> ~/.bashrc
