# syntax=docker/dockerfile:1
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
    OCR_VENV_PATH=/opt/venvs/ocr \
    OCR_PYTHON=/opt/venvs/ocr/bin/python \
    NDLOCR_VENV_PATH=/opt/venvs/ndlocr \
    NDLOCR_PYTHON=/opt/venvs/ndlocr/bin/python \
    NDLOCR_DEVICE=cpu \
    PADDLE_WHEEL_INDEX=cu126 \
    BASE_IMAGE_CUDA=12.8 \
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
    sox \
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

# ── Python dependencies (single venv = system site-packages) ─────────────────
WORKDIR ${APP_DIR}
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt
RUN python3 -c "import qwen_tts, torch, transformers; print('build preflight: qwen_tts/torch/transformers ok')"

# ── PaddleOCR dedicated venv (GPU) ───────────────────────────────────────────
# Base image is CUDA 12.8, but Paddle is installed from the official cu126
# wheel index because cu128 is not a documented stable install target in the
# current Paddle install guide.
RUN python3 -m venv /opt/venvs/ocr \
    && /opt/venvs/ocr/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/venvs/ocr/bin/python -m pip uninstall -y paddlepaddle paddlepaddle-gpu || true \
    && /opt/venvs/ocr/bin/python -m pip cache purge || true \
    && /opt/venvs/ocr/bin/python -m pip install --no-cache-dir \
        paddlepaddle-gpu==3.2.2 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    && /opt/venvs/ocr/bin/python -m pip install --no-cache-dir \
        paddleocr==3.4.0 \
        paddlex==3.4.3

# ── NDLOCR-Lite (vendored upstream checkout at fixed commit) ─────────────────
ARG NDLOCR_LITE_REPO=https://github.com/ndl-lab/ndlocr-lite.git
ARG NDLOCR_LITE_COMMIT=master
ARG NDLOCR_DEVICE=cpu
ENV NDLOCR_DEVICE=${NDLOCR_DEVICE}
RUN set -eux; \
    commit="${NDLOCR_LITE_COMMIT:-master}"; \
    if [ -z "${commit}" ]; then commit=master; fi; \
    echo "NDLOCR_LITE_REPO=${NDLOCR_LITE_REPO}"; \
    echo "NDLOCR_LITE_COMMIT_RESOLVED=${commit}"; \
    printf '%s' "${commit}" > /tmp/ndlocr_commit.txt
RUN set -eux; \
    commit="$(cat /tmp/ndlocr_commit.txt)"; \
    rm -rf /opt/ndlocr-lite; \
    git clone --filter=blob:none "${NDLOCR_LITE_REPO}" /opt/ndlocr-lite; \
    git -C /opt/ndlocr-lite checkout "${commit}"; \
    git -C /opt/ndlocr-lite rev-parse HEAD
RUN set -eux; \
    test -f /opt/ndlocr-lite/src/ocr.py; \
    find /opt/ndlocr-lite/src/model -maxdepth 1 -type f -name '*.onnx' -print; \
    test "$(find /opt/ndlocr-lite/src/model -maxdepth 1 -type f -name '*.onnx' | wc -l)" -eq 4; \
    ls -la /opt/ndlocr-lite/src/config; \
    test -f /opt/ndlocr-lite/src/config/ndl.yaml; \
    test -f /opt/ndlocr-lite/src/config/NDLmoji.yaml
RUN <<'SH'
set -eux

python3 -m venv /opt/venvs/ndlocr
/opt/venvs/ndlocr/bin/python -m pip install --upgrade pip setuptools wheel
/opt/venvs/ndlocr/bin/python -m pip install --no-cache-dir -r /opt/ndlocr-lite/requirements.txt

if [ "${NDLOCR_DEVICE}" = "cuda" ]; then
  /opt/venvs/ndlocr/bin/python -m pip uninstall -y onnxruntime || true
  /opt/venvs/ndlocr/bin/python -m pip install --no-cache-dir onnxruntime-gpu==1.23.2
fi

/opt/venvs/ndlocr/bin/python - <<'PY'
import onnxruntime
import yaml
import cv2
import numpy
print("onnxruntime ok")
PY
SH

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
COPY docker/prepare_assets.py ${APP_DIR}/docker/prepare_assets.py
COPY docker/start_services.sh ${APP_DIR}/docker/start_services.sh
RUN chmod +x /entrypoint.sh ${APP_DIR}/docker/start_services.sh

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
