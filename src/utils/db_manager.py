"""
Database management for change tracking.

Uses SQLite via SQLAlchemy to track document versions,
detected changes, and alert history.
"""

from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from src.utils.config import settings
from src.utils.logger import logger


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class Document(Base):
    """Tracks ingested documents and their versions."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_url = Column(String(500), nullable=True)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False)
    version = Column(Integer, default=1)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(DateTime, nullable=True)


class ChangeRecord(Base):
    """Records detected changes in visa documents/policies."""
    __tablename__ = "changes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, nullable=True)
    severity = Column(String(20), nullable=False)  # CRITICAL, IMPORTANT, MINOR
    change_type = Column(String(100), nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)
    source_url = Column(String(500), nullable=True)
    notified = Column(Integer, default=0)  # 0=no, 1=pending, 2=sent


class AlertLog(Base):
    """Tracks alert delivery history."""
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    change_id = Column(Integer, nullable=True)
    channel = Column(String(50), nullable=False)  # discord, email
    status = Column(String(20), nullable=False)  # sent, failed
    sent_at = Column(DateTime, default=datetime.utcnow)
    error_message = Column(Text, nullable=True)


class DatabaseManager:
    """Manages SQLite database connections and operations."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(settings.changes_db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created at {}", self.db_path)

    def get_session(self):
        """Get a new database session."""
        return self.SessionLocal()

    def add_document(self, file_path: str, file_hash: str,
                     source_url: str | None = None) -> Document:
        """Add or update a document record."""
        with self.get_session() as session:
            # Check if document already exists
            existing = (
                session.query(Document)
                .filter(Document.file_path == file_path)
                .first()
            )
            if existing:
                existing.file_hash = file_hash
                existing.version += 1
                existing.ingested_at = datetime.utcnow()
                session.commit()
                logger.info("Document updated: {} (v{})", file_path, existing.version)
                return existing
            else:
                doc = Document(
                    file_path=file_path,
                    file_hash=file_hash,
                    source_url=source_url,
                )
                session.add(doc)
                session.commit()
                logger.info("Document added: {}", file_path)
                return doc

    def record_change(
        self,
        severity: str,
        change_type: str,
        old_value: str | None = None,
        new_value: str | None = None,
        summary: str | None = None,
        source_url: str | None = None,
        document_id: int | None = None,
    ) -> ChangeRecord:
        """Record a detected change."""
        with self.get_session() as session:
            change = ChangeRecord(
                document_id=document_id,
                severity=severity,
                change_type=change_type,
                old_value=old_value,
                new_value=new_value,
                summary=summary,
                source_url=source_url,
            )
            session.add(change)
            session.commit()
            logger.info("Change recorded: [{}] {}", severity, change_type)
            return change

    def get_recent_changes(self, limit: int = 20) -> list[ChangeRecord]:
        """Get the most recent changes."""
        with self.get_session() as session:
            return (
                session.query(ChangeRecord)
                .order_by(ChangeRecord.detected_at.desc())
                .limit(limit)
                .all()
            )