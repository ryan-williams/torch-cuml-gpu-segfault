ARG CUDA_VERSION_FULL=11.6.1
FROM nvidia/cuda:$CUDA_VERSION_FULL-base-ubuntu20.04

ENV TZ=America/New_York DEBIAN_FRONTEND=noninteractive USERNAME=user

ARG CUDA_VERSION_MINOR=11.6
ARG CUDA_VERSION_FULL=11.6.1
ENV CUDA_VERSION_MINOR=$CUDA_VERSION_MINOR CUDA_VERSION_FULL=$CUDA_VERSION_FULL

RUN apt-get update \
 && apt-get install -y g++ wget \
 && apt-get clean

WORKDIR ..

ENV CONDA_PIP=/opt/conda/bin/pip
ENV CONDA_HOME="/opt/conda"
ENV PATH="$CONDA_HOME/bin:$PATH"

ARG PYTHON_VERSION_SHORT=39
ARG PYTHON_VERSION_FULL=3.9.13
ARG CONDA_VERSION=4.12.0
ARG MAMBA_VERSION=0.24.0
RUN wget -q "https://repo.anaconda.com/miniconda/Miniconda3-py${PYTHON_VERSION_SHORT}_${CONDA_VERSION}-Linux-x86_64.sh" -O ~/miniconda.sh \
 && /bin/bash ~/miniconda.sh -b -p /opt/conda \
 && rm ~/miniconda.sh \
 && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
 && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
 && echo "conda activate base" >> ~/.bashrc \
 && /opt/conda/bin/conda install -c conda-forge -y "conda=${CONDA_VERSION}" "python=${PYTHON_VERSION_FULL}" "mamba=${MAMBA_VERSION}" pip \
 && /opt/conda/bin/conda clean -afy

WORKDIR src

COPY environment.yml environment.yml
RUN mamba env update -n base -f environment.yml \
 && mamba clean -afy

COPY metaflow .metaflow
COPY pipeline.py pipeline.py
COPY entrypoint.sh entrypoint.sh

ENV PYTHONFAULTHANDLER=1
ENTRYPOINT [ "./entrypoint.sh" ]
