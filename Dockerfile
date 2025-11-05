# Use miniconda base image with Python 3.11
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl https://sdk.cloud.google.com | bash && \
    export PATH=$PATH:/root/google-cloud-sdk/bin && \
    gcloud components install gsutil -q

# Add gcloud and conda to PATH
ENV PATH="/root/google-cloud-sdk/bin:/opt/conda/bin:${PATH}"

# Copy pyproject.toml first for better caching
COPY pyproject.toml ./

# Copy the entire project
COPY . .

# Install tempus_bench package in editable mode (this installs all dependencies from pyproject.toml)
RUN pip install --no-cache-dir -e .

# Set permissions for entrypoint script (already copied above)
RUN chmod +x /app/docker-entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

