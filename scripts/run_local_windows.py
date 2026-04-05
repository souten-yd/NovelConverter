from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_FILE = ROOT / "runtime" / "local_windows_status.json"


def _load_bootstrap_state() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    return {"first_run": False, "reused_env": True}


def _runtime_status() -> dict:
    from app.shared.runtime_backends import detect_runtime_backends

    status = detect_runtime_backends()
    bootstrap = _load_bootstrap_state()
    status["first_run"] = bool(bootstrap.get("first_run", False))
    status["reused_env"] = bool(bootstrap.get("reused_env", True))
    return status


def _build_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["DATA_DIR"] = str(ROOT / "data")
    env["MODELS_ROOT"] = str(ROOT / "data" / "models")
    env["CACHE_ROOT"] = str(ROOT / "cache")
    env["HF_HOME"] = str(ROOT / "cache" / "huggingface")
    env.setdefault("TTS_BASE_USE_REAL", "true")
    env.setdefault("TTS_CUSTOM_USE_REAL", "true")
    env.setdefault("TTS_DESIGN_USE_REAL", "true")
    env.setdefault("OCR_PADDLE_DEVICE", "cpu")
    env.setdefault("PYTHONUTF8", "1")

    status = _runtime_status()
    env["NOVELCONVERTER_RUNTIME_STATUS"] = json.dumps(status, ensure_ascii=False)

    # LLM backend: Vulkan first, CPU fallback
    if status["selected_llm_backend"] == "vulkan":
        vulkan_bin = ROOT / "runtime" / "llama" / "windows-vulkan" / "llama-server.exe"
        if vulkan_bin.exists():
            env["LLAMA_SERVER_BIN"] = str(vulkan_bin)
    return env


def _spawn(cmd: list[str], env: dict) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def main() -> int:
    if os.name != "nt":
        print("[run_local_windows] WARNING: intended for Windows")

    env = _build_env()
    py = Path(sys.executable)
    procs = [
        _spawn([str(py), "-m", "uvicorn", "app.workers.tts_base.main:app", "--host", "0.0.0.0", "--port", "8001"], env),
        _spawn([str(py), "-m", "uvicorn", "app.workers.tts_custom.main:app", "--host", "0.0.0.0", "--port", "8002"], env),
        _spawn([str(py), "-m", "uvicorn", "app.workers.tts_design.main:app", "--host", "0.0.0.0", "--port", "8003"], env),
        _spawn([str(py), "-m", "uvicorn", "app.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"], env),
    ]

    print("[run_local_windows] Services started:")
    print("  Orchestrator: http://localhost:8000")
    print("  TTS Base:     http://localhost:8001")
    print("  TTS Custom:   http://localhost:8002")
    print("  TTS Design:   http://localhost:8003")

    def _shutdown(*_args):
        for p in procs:
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    rc = 0
    try:
        for p in procs:
            p.wait()
            if p.returncode and rc == 0:
                rc = p.returncode
    finally:
        _shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
