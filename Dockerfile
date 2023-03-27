ARG CUDA_VERSION=11.6.1
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y g++ wget \
 && apt-get clean

WORKDIR ..

ENV PYTHONFAULTHANDLER=1 USERNAME=user PATH="/opt/conda/bin:$PATH"

RUN wget -q "https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh" -O ~/miniconda.sh \
 && /bin/bash ~/miniconda.sh -b -p /opt/conda \
 && rm ~/miniconda.sh \
 && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
 && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
 && echo "conda activate base" >> ~/.bashrc \
 && conda install -c conda-forge -y "conda=4.12.0" "python=3.9.13" "mamba=0.24.0" pip \
 && conda clean -afy

WORKDIR src

COPY environment.yml environment.yml
RUN mamba env update -n base -f environment.yml \
 && mamba clean -afy

COPY .metaflow-example .metaflow-example
COPY pipeline.py pipeline.py
COPY entrypoint.sh entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]
