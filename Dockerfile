# =============================================================================
# NovelConverter – RunPod / NVIDIA GPU Docker Image
# Base: Ubuntu 22.04 + CUDA 12.1 + cuDNN 8
# All services run in a single container managed by supervisord
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── Build args ────────────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.11
ARG APP_DIR=/app
ARG DATA_DIR=/workspace/data

# ── Environment ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # App paths
    APP_DIR=${APP_DIR} \
    DATA_DIR=${DATA_DIR} \
    # Service ports (RunPod exposes 8000 as primary)
    ORCHESTRATOR_PORT=8000 \
    TTS_BASE_PORT=8001 \
    TTS_CUSTOM_PORT=8002 \
    TTS_DESIGN_PORT=8003 \
    # Worker URLs (intra-container localhost)
    TTS_BASE_URL=http://localhost:8001 \
    TTS_CUSTOM_URL=http://localhost:8002 \
    TTS_DESIGN_URL=http://localhost:8003 \
    # Database
    DATABASE_URL=sqlite:////workspace/data/novelconverter.db \
    # TTS mode: set to "true" to load real Qwen3-TTS model
    TTS_BASE_USE_REAL=false \
    TTS_CUSTOM_USE_REAL=false \
    TTS_DESIGN_USE_REAL=false \
    # Model paths (override to use local weights)
    TTS_BASE_MODEL_PATH=Qwen/Qwen3-TTS \
    TTS_CUSTOM_MODEL_PATH=Qwen/Qwen3-TTS \
    TTS_DESIGN_MODEL_PATH=Qwen/Qwen3-TTS \
    # LLM (optional – for speaker segmentation)
    LLM_API_URL="" \
    LLM_API_KEY="" \
    LLM_MODEL=gpt-4o-mini \
    LLAMA_SERVER_BIN=/opt/llama-cpp/bin/llama-server \
    # HuggingFace cache → /workspace for RunPod persistence
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache \
    LD_LIBRARY_PATH=/opt/llama-cpp/lib:${LD_LIBRARY_PATH}

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    git \
    jq \
    tar \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── llama.cpp server binary (prebuilt CUDA artifact) ─────────────────────────
RUN set -eux; \
    release_json="/tmp/llama_release.json"; \
    asset_regex='^llama\.cpp-b[0-9]+-cuda-12\.1\.tar\.gz$'; \
    curl -fsSL "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest" -o "${release_json}"; \
    asset_url="$(jq -r --arg re "${asset_regex}" '.assets[] | select(.name | test($re)) | .browser_download_url' "${release_json}" | head -n1)"; \
    asset_name="$(jq -r --arg re "${asset_regex}" '.assets[] | select(.name | test($re)) | .name' "${release_json}" | head -n1)"; \
    test -n "${asset_url}"; \
    test "${asset_url}" != "null"; \
    curl -fL "${asset_url}" -o "/tmp/${asset_name}"; \
    mkdir -p /tmp/llama_extract /opt/llama-cpp/bin /opt/llama-cpp/lib; \
    tar -xzf "/tmp/${asset_name}" -C /tmp/llama_extract; \
    source_root="$(dirname "$(find /tmp/llama_extract -type f -name llama-server -perm -u+x | head -n1)")"; \
    test -n "${source_root}"; \
    cp -a "${source_root}/llama-server" /opt/llama-cpp/bin/llama-server; \
    if [ -f "${source_root}/llama-cli" ]; then cp -a "${source_root}/llama-cli" /opt/llama-cpp/bin/llama-cli; fi; \
    find "${source_root}" \( -type f -o -type l \) -name '*.so*' -exec cp -a {} /opt/llama-cpp/lib/ \;; \
    rm -rf /tmp/llama_extract "${release_json}" "/tmp/${asset_name}"

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1 \
    && python3 -m pip install --upgrade pip

# ── Python dependencies (single venv = system site-packages) ─────────────────
WORKDIR ${APP_DIR}
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY app/        ${APP_DIR}/app/
COPY pyproject.toml ${APP_DIR}/
COPY samples/    ${APP_DIR}/samples/

# Make app importable without editable install
ENV PYTHONPATH=${APP_DIR}

# ── Supervisor configuration ──────────────────────────────────────────────────
COPY docker/supervisord.conf /etc/supervisor/conf.d/novelconverter.conf

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Runtime directories ───────────────────────────────────────────────────────
# /workspace is RunPod's persistent volume mount point
RUN mkdir -p \
    /workspace/data/projects \
    /workspace/data/outputs \
    /workspace/data/temp \
    /workspace/data/references \
    /workspace/hf_cache \
    /var/log/novelconverter

# ── Health check (orchestrator) ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Expose ports ──────────────────────────────────────────────────────────────
EXPOSE 8000 8001 8002 8003

ENTRYPOINT ["/entrypoint.sh"]
