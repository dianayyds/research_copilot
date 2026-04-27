from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def _sqlite_path(url: str) -> Path | None:
    return Path(url.replace("sqlite:///", "", 1)).resolve() if url.startswith("sqlite:///") else None


def _engine_kwargs(url: str) -> dict:
    return {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}


database_url = settings.resolved_database_url
sqlite_path = _sqlite_path(database_url)
if sqlite_path:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(database_url, pool_pre_ping=True, future=True, **_engine_kwargs(database_url))
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


async def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app import db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    if database_url.startswith("mysql"):
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE assets MODIFY COLUMN content LONGTEXT NOT NULL"))
