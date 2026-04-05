# =============================================================================
# NovelConverter – RunPod / NVIDIA GPU Docker Image
# Base: Ubuntu 22.04 + CUDA 12.8 + cuDNN
# All services run in a single container managed by supervisord
# =============================================================================
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# ── Build args ────────────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.11
ARG APP_DIR=/workspace/NovelConverter
ARG DATA_DIR=/workspace/data

# ── Environment ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
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
    TTS_BASE_USE_REAL=true \
    TTS_CUSTOM_USE_REAL=true \
    TTS_DESIGN_USE_REAL=true \
    # Model paths – correct per-mode Qwen3-TTS variants
    TTS_BASE_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    TTS_CUSTOM_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    TTS_DESIGN_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    TTS_TOKENIZER_PATH=Qwen/Qwen3-TTS-Tokenizer-12Hz \
    # LLM (optional – for speaker segmentation)
    LLM_API_URL="" \
    LLM_API_KEY="" \
    LLM_MODEL=gpt-4o-mini \
    LLAMA_SERVER_BIN=/opt/llama-cpp/bin/llama-server \
    # HuggingFace cache → /workspace for RunPod persistence
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache \
    # Skip PaddleX online model-source probe to reduce cold-start latency
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    LD_LIBRARY_PATH=/opt/llama-cpp/lib:${LD_LIBRARY_PATH}

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    git \
    tar \
    ffmpeg \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-jpn \
    tesseract-ocr-jpn-vert \
    tesseract-ocr-eng \
    # Archive support
    unrar-free \
    p7zip-full \
    libarchive-tools \
    # Audio
    libsndfile1 \
    libsndfile1-dev \
    # Build tools
    build-essential \
    # Python
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    # Process manager
    supervisor \
    # OpenCV / PaddleOCR dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── llama.cpp server binary (prebuilt CUDA artifact) ─────────────────────────
RUN set -eux; \
    archive_url="https://github.com/ai-dock/llama.cpp-cuda/releases/download/b8642/llama.cpp-b8642-cuda-12.8.tar.gz"; \
    archive_path="/tmp/llama.cpp-b8642-cuda-12.8.tar.gz"; \
    extract_dir="/tmp/llama-prebuilt"; \
    curl -fL "${archive_url}" -o "${archive_path}"; \
    mkdir -p "${extract_dir}" /opt/llama-cpp/bin /opt/llama-cpp/lib; \
    tar -xzf "${archive_path}" -C "${extract_dir}"; \
    echo "=== extracted files (maxdepth 4) ==="; \
    find "${extract_dir}" -maxdepth 4 -type f | sort; \
    echo "=== searching for llama-server ==="; \
    llama_path="$(find "${extract_dir}" -type f -name 'llama-server' | head -n1)"; \
    if [ -z "${llama_path}" ]; then \
      echo "ERROR: llama-server not found in extracted archive"; \
      find "${extract_dir}" -maxdepth 6 | sort; \
      exit 1; \
    fi; \
    source_root="$(dirname "${llama_path}")"; \
    echo "llama_path=${llama_path}"; \
    echo "source_root=${source_root}"; \
    test -f "${source_root}/llama-server"; \
    cp -a "${source_root}/llama-server" /opt/llama-cpp/bin/llama-server; \
    if [ -f "${source_root}/llama-cli" ]; then cp -a "${source_root}/llama-cli" /opt/llama-cpp/bin/llama-cli; fi; \
    find "${source_root}" \( -type f -o -type l \) -name '*.so*' -exec cp -a {} /opt/llama-cpp/lib/ \; || true; \
    chmod +x /opt/llama-cpp/bin/llama-server; \
    ls -l /opt/llama-cpp/bin/llama-server; \
    ldd /opt/llama-cpp/bin/llama-server || true; \
    rm -rf "${extract_dir}" "${archive_path}"

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1 \
    && python3 -m pip install --upgrade pip

# ── PaddleOCR GPU backend ─────────────────────────────────────────────────────
# Pinned to 3.3.1 (tested with paddleocr==3.4.0, paddlex==3.4.3).
# Prefer CUDA build on RunPod GPU image. If GPU wheel resolution fails,
# fall back to CPU build so the container can still boot.
RUN pip install --no-cache-dir "paddlepaddle-gpu==3.3.1" \
      -f https://www.paddlepaddle.org.cn/packages/stable/cu128/ \
    || pip install --no-cache-dir "paddlepaddle==3.3.1"

# ── Python dependencies (single venv = system site-packages) ─────────────────
WORKDIR ${APP_DIR}
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# ── NDLOCR-Lite (vendored upstream checkout at fixed commit) ─────────────────
ARG NDLOCR_LITE_REPO=https://github.com/ndl-lab/ndlocr-lite.git
ARG NDLOCR_LITE_COMMIT
RUN test -n "${NDLOCR_LITE_COMMIT}" \
    && git clone --filter=blob:none ${NDLOCR_LITE_REPO} /opt/ndlocr-lite \
    && git -C /opt/ndlocr-lite checkout ${NDLOCR_LITE_COMMIT} \
    && test -f /opt/ndlocr-lite/src/ocr.py \
    && test "$(find /opt/ndlocr-lite/src/model -maxdepth 1 -type f -name '*.onnx' | wc -l)" -eq 4 \
    && test -f /opt/ndlocr-lite/src/config/ndl.yaml \
    && test -f /opt/ndlocr-lite/src/config/NDLmoji.yaml

# ── Application code ──────────────────────────────────────────────────────────
COPY app/        ${APP_DIR}/app/
COPY scripts/    ${APP_DIR}/scripts/
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
