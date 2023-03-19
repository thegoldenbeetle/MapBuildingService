FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
        python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        ## Python
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /source
COPY pyproject.toml .
RUN python3 -m pip install -U pip wheel pip-tools
RUN pip-compile --extra dev --resolver=backtracking -o requirements.txt pyproject.toml
RUN python3 -m pip install -r requirements.txt
