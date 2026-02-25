"""
db.py — Supabase PostgreSQL metadata registry.

Replaces registry.json with a proper relational table.
Stores document metadata (filename, pages, chunks, upload time)
per index_id so filenames survive server restarts and are
accessible across multiple workers.

Table: document_registry
  index_id    TEXT PRIMARY KEY
  filename    TEXT
  total_pages INTEGER
  total_chunks INTEGER
  doc_type    TEXT  ('document' | 'collection')
  uploaded_at TIMESTAMP
  extra       JSONB  (collection file list etc.)
"""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, text,
    Column, String, Integer, DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DATABASE_URL, DB_ENABLED

Base = declarative_base()


#  ORM Model 

class DocumentRegistry(Base):
    __tablename__ = "document_registry"

    index_id     = Column(String, primary_key=True)
    filename     = Column(String, nullable=False)
    total_pages  = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    doc_type     = Column(String, default="document")
    uploaded_at  = Column(DateTime, default=datetime.utcnow)
    extra        = Column(Text, default="{}")   # JSON string for flexibility


#  Engine setup 

_engine  = None
_Session = None


def get_engine():
    global _engine
    if _engine is None and DB_ENABLED:
        try:
            _engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                connect_args={"connect_timeout": 10},
            )
        except Exception as e:
            print(f"⚠ Could not create DB engine: {e}")
            _engine = None
    return _engine


def init_db():
    """Create table if it doesn't exist. Called once at startup."""
    engine = get_engine()
    if not engine:
        print("⚠ DATABASE_URL not set — falling back to registry.json")
        return
    try:
        Base.metadata.create_all(engine)
        print("✓ Supabase PostgreSQL connected — document_registry ready")
    except Exception as e:
        print(f"⚠ Supabase connection failed: {e}")
        print("⚠ Falling back to registry.json — check DATABASE_URL in .env")


def get_session():
    global _Session
    if _Session is None:
        engine = get_engine()
        if engine:
            _Session = sessionmaker(bind=engine)
    return _Session() if _Session else None


#  Registry operations 

def save_document(index_id: str, filename: str, total_pages: int,
                  total_chunks: int, doc_type: str = "document",
                  extra: dict = None):
    """Insert or update a document record."""
    session = get_session()
    if not session:
        return

    try:
        record = DocumentRegistry(
            index_id     = index_id,
            filename     = filename,
            total_pages  = total_pages,
            total_chunks = total_chunks,
            doc_type     = doc_type,
            uploaded_at  = datetime.utcnow(),
            extra        = json.dumps(extra or {}),
        )
        session.merge(record)   # upsert — safe to call multiple times
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"DB write error: {e}")
    finally:
        session.close()


def get_all_documents() -> list:
    """Return all documents sorted by upload time descending."""
    session = get_session()
    if not session:
        return []

    try:
        rows = session.query(DocumentRegistry)\
                      .order_by(DocumentRegistry.uploaded_at.desc())\
                      .all()
        return [
            {
                "index_id":     r.index_id,
                "filename":     r.filename,
                "total_pages":  r.total_pages,
                "total_chunks": r.total_chunks,
                "doc_type":     r.doc_type,
                "uploaded_at":  r.uploaded_at.isoformat() if r.uploaded_at else None,
                "extra":        json.loads(r.extra or "{}"),
            }
            for r in rows
        ]
    except Exception as e:
        print(f"DB read error: {e}")
        return []
    finally:
        session.close()


def get_document(index_id: str) -> Optional[dict]:
    """Fetch a single document record by index_id."""
    session = get_session()
    if not session:
        return None

    try:
        row = session.query(DocumentRegistry)\
                     .filter_by(index_id=index_id)\
                     .first()
        if not row:
            return None
        return {
            "index_id":     row.index_id,
            "filename":     row.filename,
            "total_pages":  row.total_pages,
            "total_chunks": row.total_chunks,
            "doc_type":     row.doc_type,
            "uploaded_at":  row.uploaded_at.isoformat() if row.uploaded_at else None,
        }
    except Exception as e:
        print(f"DB read error: {e}")
        return None
    finally:
        session.close()