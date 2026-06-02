# Use an official Python image as the base
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates time \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . /app

# Sanity-check: ensure bundled FeynGym sources are present
RUN test -f FeynGym/pyfeyngym/install_julia_packages.py || \
    (echo "ERROR: bundled FeynGym sources are missing from the Docker build context" && \
     exit 1)

# Install Python dependencies

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        JULIA_URL="https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.6-linux-x86_64.tar.gz"; \
        JULIA_DIR="julia-1.12.6"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        JULIA_URL="https://julialang-s3.julialang.org/bin/linux/aarch64/1.12/julia-1.12.6-linux-aarch64.tar.gz"; \
        JULIA_DIR="julia-1.12.6"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    curl -fsSL "$JULIA_URL" -o julia.tar.gz && \
    tar -xzf julia.tar.gz && \
    mv "$JULIA_DIR" /opt/julia && \
    ln -s /opt/julia/bin/julia /usr/local/bin/julia && \
    rm julia.tar.gz
RUN cd FeynGym/pyfeyngym && pip install --no-cache-dir --root-user-action=ignore -e .
RUN python FeynGym/pyfeyngym/install_julia_packages.py

RUN cd FeynGym/ppo_masking && pip install --no-cache-dir --root-user-action=ignore -e .

RUN pip install --no-cache-dir torch cma notebook pulp

RUN curl -L "https://kira.hepforge.org/downloads?f=binaries/kira-3.1" -o /usr/local/bin/kira \
    && chmod +x /usr/local/bin/kira

# Download and install Fermat (required by kira)
RUN curl -sL -o /tmp/Ferl7.tar.gz "https://home.bway.net/lewis/fermat64/Ferl7.tar.gz" \
    && mkdir -p /opt/fermat \
    && tar -xzf /tmp/Ferl7.tar.gz -C /opt/fermat --strip-components=1 \
    && rm /tmp/Ferl7.tar.gz

# Set FERMATPATH so kira can find the Fermat binary
ENV FERMATPATH=/opt/fermat/fer64
