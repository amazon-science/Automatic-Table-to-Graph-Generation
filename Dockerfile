FROM nvidia/cuda:11.7.1-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    unzip \
    curl \
    wget \
    bzip2 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Initialize conda
RUN conda init bash

# Accept conda Terms of Service to avoid interactive prompts
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Set working directory
WORKDIR /workspace

# Copy the project files
COPY . /workspace/

# (optional) show conda version for debugging
RUN /opt/conda/bin/conda --version

# Create conda environment (GPU 11.7 / Python 3.9 / Torch 1.13.1)
# Add a cache-buster tied to the template so this step rebuilds when env spec changes
RUN cd /workspace/multi-table-benchmark \
 && sha256sum conda/env.yml.template \
 && bash conda/create_conda_env.sh -s -g 11.7 -p 3.9 -t 1.13.1 \
 && /opt/conda/bin/conda info --envs \
 && test -d /opt/conda/envs/dbinfer-gpu

# Install extra pip packages inside the env (no shell activation games)
RUN /opt/conda/bin/conda run -n dbinfer-gpu pip install \
    codetiming \
    humanfriendly \
    sentence_transformers==3.3.0 \
    transformers==4.44.2 \
    nltk==3.9.1

# Clone DeepJoin repository (skip if already exists)
RUN if [ ! -d "/workspace/deepjoin" ] || [ -z "$(ls -A /workspace/deepjoin)" ]; then \
        git clone https://github.com/mutong184/deepjoin /workspace/deepjoin; \
    else \
        echo "DeepJoin directory already exists and is not empty, skipping clone"; \
    fi

# Set environment variables for automatic conda activation
ENV CONDA_DEFAULT_ENV=dbinfer-gpu
ENV CONDA_PREFIX=/opt/conda/envs/dbinfer-gpu
ENV PATH="/opt/conda/envs/dbinfer-gpu/bin:${PATH}"

# Create a startup script that activates conda environment (place outside /opt)
RUN echo '#!/bin/bash' > /usr/local/bin/startup.sh && \
    echo 'source /opt/conda/bin/activate dbinfer-gpu' >> /usr/local/bin/startup.sh && \
    echo 'cd /workspace' >> /usr/local/bin/startup.sh && \
    echo 'exec "$@"' >> /usr/local/bin/startup.sh && \
    chmod +x /usr/local/bin/startup.sh

# Set the startup script as entrypoint
ENTRYPOINT ["/usr/local/bin/startup.sh"]
CMD ["bash"]

