from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

_DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "novelconverter.db"
DB_URL = os.environ.get("DATABASE_URL", f"sqlite:///{_DEFAULT_DB}")

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
