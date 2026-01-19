FROM nvidia/cuda:13.0.1-base-ubuntu22.04

ENV PYTHON_VERSION=3.11
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install dependencies and clean up in the same layer
# hadolint ignore=DL3008
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y update \
    && apt-get -y install --no-install-recommends \
    python3.11=3.11.0~rc1-1~22.04 \
    git \
    ffmpeg=7:4.4.2-0ubuntu0.22.04.1 \
    libcudnn9-cuda-12=9.8.0.87-1 \
    libatomic1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install UV for package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create cache directories that are not mounted as volumes
RUN mkdir -p /var/cache/uv /var/cache/pip

# Configure UV to use cache directory that is not mounted as volume
# Use /var/cache/uv instead of /tmp to avoid conflicts with volume mounts
ENV UV_CACHE_DIR=/var/cache/uv
ENV UV_LINK_MODE=copy
ENV PIP_CACHE_DIR=/var/cache/pip

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY app app/
COPY tests tests/
COPY scripts scripts/
COPY app/gunicorn_logging.conf .

# Copy local torch and torchaudio wheel files if they exist
COPY torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl* ./
COPY torchaudio-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl* ./

# Install Python dependencies using UV with pyproject.toml
# UV automatically selects CUDA 12.8 wheels on Linux
# Temporarily remove torch and torchaudio from pyproject.toml, sync dependencies, then install local wheels
# Clean up UV cache after each major step to prevent disk space issues
RUN if [ -f torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl ] || [ -f torchaudio-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl ]; then \
        cp pyproject.toml pyproject.toml.bak \
        && sed -i '/torch<=2.8.0/d' pyproject.toml \
        && sed -i '/torchaudio<=2.8.0/d' pyproject.toml \
        && uv sync --frozen --no-dev \
        && rm -rf /var/cache/uv /root/.cache/uv \
        && if [ -f torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl ]; then \
            uv pip install --system ./torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl \
            && rm -rf /var/cache/uv /root/.cache/uv; \
        fi \
        && if [ -f torchaudio-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl ]; then \
            uv pip install --system ./torchaudio-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl \
            && rm -rf /var/cache/uv /root/.cache/uv; \
        fi \
        && mv pyproject.toml.bak pyproject.toml; \
    else \
        uv sync --frozen --no-dev \
        && rm -rf /var/cache/uv /root/.cache/uv; \
    fi \
    && uv pip install --system ctranslate2==4.6.0 \
    && rm -rf /var/cache/uv /root/.cache/uv /root/.uv /var/cache/pip \
    && rm -rf /root/.cache/pip /root/.cache/huggingface torch-*.whl torchaudio-*.whl \
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.pyo' -delete \
    && find /app -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /app -type f -name '*.pyc' -delete

EXPOSE 8000

# Make entrypoint script executable
RUN chmod +x scripts/docker-entrypoint.sh

# Health check to verify the application is responsive
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

ENTRYPOINT ["/bin/bash", "scripts/docker-entrypoint.sh"]
