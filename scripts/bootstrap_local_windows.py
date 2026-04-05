from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
CONFIG_PATH = ROOT / "config" / "runtime_windows_local.json"
REQ_FILE = ROOT / "requirements-local-windows.txt"


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("[bootstrap]$", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), check=check)


def _python311_ok() -> bool:
    return sys.version_info.major == 3 and sys.version_info.minor == 11


def _ensure_venv() -> tuple[Path, bool]:
    py = VENV / "Scripts" / "python.exe"
    if py.exists():
        return py, False
    _run([sys.executable, "-m", "venv", str(VENV)])
    return py, True


def _ensure_pip_and_deps(py: Path) -> None:
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _run([str(py), "-m", "pip", "install", "-r", str(REQ_FILE)])


def _ensure_dirs(config: dict[str, Any]) -> None:
    dirs = [
        ROOT / ".venv",
        ROOT / config["models_root"],
        ROOT / config["cache_root"] / "huggingface",
        ROOT / "runtime" / "llama" / "windows-vulkan",
        ROOT / "data" / "models" / "qwen3_tts",
        ROOT / "data" / "models" / "llm",
        ROOT / "data" / "models" / "ocr" / "ndlocr",
        ROOT / "data" / "models" / "ocr" / "paddlex",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _ensure_llama_runtime(py: Path, config: dict[str, Any]) -> None:
    runtime_dir = ROOT / config["llama"]["runtime_dir"]
    binary = runtime_dir / "llama-server.exe"
    if binary.exists():
        print(f"[bootstrap] llama.cpp runtime already present: {binary}")
        return

    print("[bootstrap] llama.cpp runtime missing, downloading official release (best effort)")
    try:
        code = (
            "from huggingface_hub import hf_hub_download;"
            "p=hf_hub_download(repo_id='ggml-org/llama.cpp',"
            "filename='llama-b4208-bin-win-vulkan-x64.zip',"
            f"local_dir=r'{runtime_dir.as_posix()}',"
            "local_dir_use_symlinks=False);"
            "print(p)"
        )
        cp = subprocess.run([str(py), "-c", code], cwd=str(ROOT), capture_output=True, text=True)
        if cp.returncode != 0:
            print(f"[bootstrap] WARNING: could not fetch Vulkan build: {cp.stderr.strip()}")
            return
        zf = Path(cp.stdout.strip().splitlines()[-1])
        if zf.is_file() and zf.suffix.lower() == ".zip":
            with zipfile.ZipFile(zf) as z:
                z.extractall(runtime_dir)
            print(f"[bootstrap] extracted llama.cpp runtime to {runtime_dir}")
    except Exception as exc:
        print(f"[bootstrap] WARNING: llama.cpp runtime setup skipped: {exc}")


def _ensure_models(py: Path) -> None:
    code = """
from app.shared.model_manager import ensure_all_models
from app.shared.llm_downloader import ensure_llm_model
from app.shared.runtime_backends import detect_runtime_backends
from app.shared.runtime_backends import save_runtime_status

print('[bootstrap] Ensuring Qwen3-TTS models...')
print(ensure_all_models())
print('[bootstrap] Ensuring LLM GGUF model...')
print(ensure_llm_model())
status = detect_runtime_backends()
save_runtime_status(status)
print('[bootstrap] runtime status=', status)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["DATA_DIR"] = str(ROOT / "data")
    env["MODELS_ROOT"] = str(ROOT / "data" / "models")
    env["CACHE_ROOT"] = str(ROOT / "cache")
    env["HF_HOME"] = str(ROOT / "cache" / "huggingface")
    subprocess.run([str(py), "-c", code], cwd=str(ROOT), env=env, check=True)


def main() -> int:
    if os.name != "nt":
        print("[bootstrap] WARNING: this launcher is designed for Windows")
    if not _python311_ok():
        print(f"[bootstrap] ERROR: Python 3.11 required. current={sys.version}")
        return 1

    cfg = _load_config()
    py, first_run = _ensure_venv()
    _ensure_pip_and_deps(py)
    _ensure_dirs(cfg)
    _ensure_llama_runtime(py, cfg)
    _ensure_models(py)

    status_path = ROOT / "runtime" / "local_windows_status.json"
    status = {
        "first_run": first_run,
        "reused_env": not first_run,
        "venv": str(VENV),
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bootstrap] done: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
