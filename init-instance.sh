#!/usr/bin/env bash

set -ex

# Install recent Conda + libmamba-solver
d=~/miniconda
wget -q "https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh" -O ~/miniconda.sh
/bin/bash ~/miniconda.sh -b -p $d
rm ~/miniconda.sh
echo ". $d/etc/profile.d/conda.sh" >> ~/.bashrc
conda activate base
conda --version
conda install -y -n base conda-libmamba-solver
conda config --set solver libmamba

# Clone this repo
git clone https://github.com/ryan-williams/torch-cuml-metaflow-gpu-segfault gpu-segfault
cd gpu-segfault

# Create conda env with necessary dependencies (see environment.yml)
conda env update -n segfault -f environment.yml
echo "conda activate segfault" >> ~/.bashrc
. ~/.bashrc
