from pathlib import Path

from app.orchestrator.services import llm_manager


def test_resolve_llama_server_bin_from_common_build_path(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLAMA_SERVER_BIN", raising=False)
    monkeypatch.setattr(llm_manager.shutil, "which", lambda _c: None)

    build_bin = tmp_path / "build" / "bin"
    build_bin.mkdir(parents=True)
    llama_server = build_bin / "llama-server"
    llama_server.write_text("#!/bin/sh\n", encoding="utf-8")

    resolved = llm_manager._resolve_llama_server_bin()
    assert resolved is not None
    assert Path(resolved).resolve() == llama_server.resolve()


def test_resolve_llama_server_bin_prefers_llama_server_bin_env(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_BIN", "/opt/custom/llama-server")
    monkeypatch.setattr(llm_manager.shutil, "which", lambda c: c if c == "/opt/custom/llama-server" else None)

    resolved = llm_manager._resolve_llama_server_bin()
    assert resolved == "/opt/custom/llama-server"
