FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG UID=1000
ARG GID=1000

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


RUN groupadd -g "${GID}" workuser\
    && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" workuser

USER workuser

ENV PATH "$PATH:/home/workuser/.local/bin"

COPY . /source
WORKDIR /source
RUN python3 -m pip install -U pip wheel
RUN python3 -m pip install -e .

