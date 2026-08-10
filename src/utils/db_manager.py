"""
Database management for change tracking.

Uses SQLite via SQLAlchemy to track document versions,
detected changes, and alert history.
"""

from datetime import datetime

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


class PageSnapshot(Base):
    """
    Stores the extracted text of each scraped page.

    This is what makes day-over-day change detection possible: the scraper
    writes a snapshot on every run, and the next run reads the previous one
    back out to diff against. Keeping it in SQLite (rather than on the
    filesystem) means `changes.db` is the *only* state that has to survive
    between CI runs.
    """
    __tablename__ = "page_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(500), nullable=False, index=True)
    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    scraped_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Manages SQLite database connections and operations."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(settings.changes_db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        # expire_on_commit=False so objects returned from these helpers stay
        # readable after their session closes. Without it, every attribute
        # access on a returned record raises DetachedInstanceError.
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

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

    def get_changes(
        self,
        severity: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ChangeRecord]:
        """Get changes with optional severity filter."""
        with self.get_session() as session:
            query = session.query(ChangeRecord)
            if severity:
                query = query.filter(ChangeRecord.severity == severity)
            return (
                query.order_by(ChangeRecord.detected_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

    def get_change_counts_by_severity(self) -> dict[str, int]:
        """Get count of changes grouped by severity."""
        from sqlalchemy import func
        with self.get_session() as session:
            rows = (
                session.query(ChangeRecord.severity, func.count(ChangeRecord.id))
                .group_by(ChangeRecord.severity)
                .all()
            )
            return dict(rows)

    def get_total_changes(self) -> int:
        """Get total number of recorded changes."""
        from sqlalchemy import func
        with self.get_session() as session:
            return session.query(func.count(ChangeRecord.id)).scalar() or 0

    # ------------------------------------------------------------------
    # Page snapshots — the state that powers change detection
    # ------------------------------------------------------------------

    def save_page_snapshot(
        self,
        url: str,
        content: str,
        content_hash: str,
        title: str | None = None,
    ) -> None:
        """Store a scraped page snapshot for future comparison."""
        with self.get_session() as session:
            session.add(PageSnapshot(
                url=url,
                title=title,
                content=content,
                content_hash=content_hash,
            ))
            session.commit()
        logger.debug("Snapshot saved for {} ({})", url, content_hash[:12])

    def get_latest_page_snapshot(self, url: str) -> dict | None:
        """
        Get the most recent stored snapshot for a URL.

        Returns a plain dict (not an ORM object) so callers never have to
        care about session lifetime.
        """
        with self.get_session() as session:
            row = (
                session.query(PageSnapshot)
                .filter(PageSnapshot.url == url)
                .order_by(PageSnapshot.scraped_at.desc(), PageSnapshot.id.desc())
                .first()
            )
            if row is None:
                return None
            return {
                "url": row.url,
                "title": row.title,
                "content": row.content,
                "content_hash": row.content_hash,
                "scraped_at": row.scraped_at,
            }

    def prune_page_snapshots(self, url: str, keep: int = 10) -> int:
        """
        Keep only the most recent `keep` snapshots for a URL.

        Full page text adds up quickly across daily runs; this keeps
        changes.db small enough to cache between CI runs.
        """
        with self.get_session() as session:
            stale = (
                session.query(PageSnapshot)
                .filter(PageSnapshot.url == url)
                .order_by(PageSnapshot.scraped_at.desc(), PageSnapshot.id.desc())
                .offset(keep)
                .all()
            )
            for row in stale:
                session.delete(row)
            session.commit()
            return len(stale)

    # ------------------------------------------------------------------
    # Alert bookkeeping
    # ------------------------------------------------------------------

    def get_unnotified_changes(
        self,
        severities: tuple[str, ...] = ("CRITICAL", "IMPORTANT"),
        limit: int = 50,
    ) -> list[ChangeRecord]:
        """Get changes matching `severities` that have not been alerted on yet."""
        with self.get_session() as session:
            return (
                session.query(ChangeRecord)
                .filter(ChangeRecord.severity.in_(severities))
                .filter(ChangeRecord.notified != 2)
                .order_by(ChangeRecord.detected_at.desc())
                .limit(limit)
                .all()
            )

    def mark_changes_notified(self, change_ids: list[int]) -> None:
        """Mark changes as successfully alerted (notified=2)."""
        if not change_ids:
            return
        with self.get_session() as session:
            (
                session.query(ChangeRecord)
                .filter(ChangeRecord.id.in_(change_ids))
                .update({ChangeRecord.notified: 2}, synchronize_session=False)
            )
            session.commit()
        logger.info("Marked {} change(s) as notified", len(change_ids))

    def log_alert(
        self,
        channel: str,
        status: str,
        change_id: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Record an alert delivery attempt."""
        with self.get_session() as session:
            session.add(AlertLog(
                change_id=change_id,
                channel=channel,
                status=status,
                error_message=error_message,
            ))
            session.commit()
        logger.info("Alert logged: {} — {}", channel, status)
