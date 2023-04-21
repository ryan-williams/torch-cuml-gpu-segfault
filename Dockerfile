ARG CUDA=11.7.1
ARG UBUNTU=22.04
FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y g++ wget \
 && apt-get clean

WORKDIR ..

ENV d=/opt/conda
ENV PYTHONFAULTHANDLER=1 USERNAME=user PATH="/opt/conda/bin:$PATH" conda="$d/bin/conda"

# Install recent Conda + libmamba-solver
RUN wget -q "https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh" -O miniconda.sh \
 && /bin/bash miniconda.sh -b -p $d \
 && rm miniconda.sh \
 && echo ". $d/etc/profile.d/conda.sh" >> ~/.bashrc \
 && $conda install -y -n base conda-libmamba-solver \
 && $conda config --set solver libmamba \
 && $conda config --set channel_priority flexible  # https://github.com/rapidsai/cuml/issues/4016

# Create conda env with necessary dependencies (see environment.yml) \
ARG ENV=segfault
COPY environment.yml environment.yml
RUN $conda env update -n $ENV \
 && echo "conda activate $ENV" >> ~/.bashrc \
 && conda clean -afy

WORKDIR src

COPY neighbors.py neighbors.py
COPY run.py run.py

ENTRYPOINT [ "./neighbors.py" ]
