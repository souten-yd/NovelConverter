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


def init_db():
    from app.shared import models  # noqa: F401 – registers models

    Base.metadata.create_all(bind=engine)
