from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

def _default_db_url() -> str:
    data_dir = os.environ.get("DATA_DIR", str(Path(__file__).parent.parent.parent / "data"))
    db_path = Path(data_dir) / "novelconverter.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"

DB_URL = os.environ.get("DATABASE_URL", "") or _default_db_url()

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _migrate_columns(eng) -> None:
    """Add new columns to existing tables if they don't exist yet (SQLite-safe)."""
    from sqlalchemy import text
    migrations = [
        ("segments", "candidates",        "TEXT DEFAULT '[]'"),
        ("segments", "evidence_spans",    "TEXT DEFAULT '[]'"),
        ("segments", "needs_review",      "INTEGER DEFAULT 0"),
        ("segments", "monologue_subtype", "TEXT"),
        ("segments", "rule_log",          "TEXT DEFAULT '[]'"),
        ("segments", "ruby_text",         "TEXT"),
        ("segments", "ruby_metadata",     "TEXT DEFAULT '[]'"),
        ("segments", "tts_text",          "TEXT"),
        ("voice_profiles", "preset_id",   "TEXT"),
    ]
    with eng.connect() as conn:
        for table, col, col_def in migrations:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}"))
                conn.commit()
            except Exception:
                pass  # column already exists – safe to ignore


def init_db():
    from app.shared import models  # noqa: F401 – registers models

    Base.metadata.create_all(bind=engine)
    _migrate_columns(engine)
