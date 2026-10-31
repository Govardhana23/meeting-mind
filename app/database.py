"""
SQLAlchemy ORM models and async engine/session factory.

Tables
------
- teams         – known team members (name, email)
- meetings      – raw + parsed transcripts
- utterances    – per-speaker segments within a meeting
- decisions     – extracted decisions with optional embedding
- action_items  – tasks with owner, deadline, status
- conflicts     – detected contradictions between decisions
- pending_actions – confirmation queue items
- reminders_log – tracks which reminders have been sent
- digests_log   – tracks digest generation
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Optional

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, ForeignKey,
    JSON, Enum as SAEnum, UniqueConstraint, Index, create_engine
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.config import settings

# ─── Base ────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


# ─── Team Members ────────────────────────────────────────────────────────────

class TeamMember(Base):
    __tablename__ = "team_members"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(300), nullable=False, unique=True)
    aliases: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")  # comma-separated
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_dt.datetime.utcnow)

    action_items = relationship("ActionItem", back_populates="owner_member", lazy="selectin")


# ─── Meetings ────────────────────────────────────────────────────────────────

class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    meeting_date: Mapped[_dt.datetime] = mapped_column(DateTime, default=_dt.datetime.utcnow)
    raw_transcript: Mapped[str] = mapped_column(Text, nullable=False)
    format: Mapped[str] = mapped_column(String(20), default="plain")  # plain | vtt | srt
    status: Mapped[str] = mapped_column(String(30), default="uploaded")  # uploaded | processing | processed | error
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_dt.datetime.utcnow)

    utterances = relationship("Utterance", back_populates="meeting", cascade="all, delete-orphan", lazy="selectin")
    decisions = relationship("Decision", back_populates="meeting", cascade="all, delete-orphan", lazy="selectin")
    action_items = relationship("ActionItem", back_populates="meeting", cascade="all, delete-orphan", lazy="selectin")


# ─── Utterances (speaker-attributed segments) ────────────────────────────────

class Utterance(Base):
    __tablename__ = "utterances"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id"), nullable=False)
    speaker: Mapped[str] = mapped_column(String(200), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # seconds
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    meeting = relationship("Meeting", back_populates="utterances")


# ─── Decisions ───────────────────────────────────────────────────────────────

class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # surrounding discussion
    topic: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    decided_by: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array of floats
    status: Mapped[str] = mapped_column(String(30), default="active")  # active | superseded | reverted
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_dt.datetime.utcnow)

    meeting = relationship("Meeting", back_populates="decisions")
    action_items = relationship("ActionItem", back_populates="decision", lazy="selectin")
    # conflicts where this decision is the NEW one
    conflicts_as_new = relationship(
        "Conflict", foreign_keys="Conflict.new_decision_id", back_populates="new_decision", lazy="selectin"
    )
    # conflicts where this decision is the OLD one
    conflicts_as_old = relationship(
        "Conflict", foreign_keys="Conflict.old_decision_id", back_populates="old_decision", lazy="selectin"
    )


# ─── Action Items ────────────────────────────────────────────────────────────

class ActionItem(Base):
    __tablename__ = "action_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id"), nullable=False)
    decision_id: Mapped[Optional[str]] = mapped_column(ForeignKey("decisions.id"), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    owner_name: Mapped[str] = mapped_column(String(200), nullable=False)
    owner_id: Mapped[Optional[str]] = mapped_column(ForeignKey("team_members.id"), nullable=True)
    deadline: Mapped[Optional[_dt.datetime]] = mapped_column(DateTime, nullable=True)
    priority: Mapped[str] = mapped_column(String(20), default="medium")  # low | medium | high | critical
    status: Mapped[str] = mapped_column(String(30), default="pending")  # pending | confirmed | in_progress | done | cancelled
    tracker_card_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    tracker_card_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_dt.datetime.utcnow)

    meeting = relationship("Meeting", back_populates="action_items")
    decision = relationship("Decision", back_populates="action_items")
    owner_member = relationship("TeamMember", back_populates="action_items")


# ─── Conflicts ───────────────────────────────────────────────────────────────

class Conflict(Base):
    __tablename__ = "conflicts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    new_decision_id: Mapped[str] = mapped_column(ForeignKey("decisions.id"), nullable=False)
    old_decision_id: Mapped[str] = mapped_column(ForeignKey("decisions.id"), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="medium")  # low | medium | high
    resolution: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="unresolved")  # unresolved | resolved | dismissed
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_dt.datetime.utcnow)

    new_decision = relationship("Decision", foreign_keys=[new_decision_id], back_populates="conflicts_as_new")
    old_decision = relationship("Decision", foreign_keys=[old_decision_id], back_populates="conflicts_as_old")


# ─── Pending Actions (confirmation queue) ────────────────────────────────────

class PendingAction(Base):
    __tablename__ = "pending_actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id"), nullable=False)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # action_type values: create_task | resolve_conflict | notify_conflict | send_reminder
    reference_id: Mapped[str] = mapped_column(String(36), nullable=False)  # ID of the related entity
    payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON blob with details
    status: Mapped[str] = mapped_column(String(30), default="pending")  # pending | approved | rejected | executed
    reviewed_at: Mapped[Optional[_dt.datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_dt.datetime.utcnow)


# ─── Reminders Log ──────────────────────────────────────────────────────────

class ReminderLog(Base):
    __tablename__ = "reminders_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    action_item_id: Mapped[str] = mapped_column(ForeignKey("action_items.id"), nullable=False)
    hours_before: Mapped[int] = mapped_column(Integer, nullable=False)
    sent_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_dt.datetime.utcnow)
    channel: Mapped[str] = mapped_column(String(30), default="email")  # email | webhook | in_app

    __table_args__ = (
        UniqueConstraint("action_item_id", "hours_before", name="uq_reminder_per_interval"),
    )


# ─── Digests Log ─────────────────────────────────────────────────────────────

class DigestLog(Base):
    __tablename__ = "digests_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    member_id: Mapped[str] = mapped_column(ForeignKey("team_members.id"), nullable=False)
    week_start: Mapped[_dt.date] = mapped_column(DateTime, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sent_at: Mapped[Optional[_dt.datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_dt.datetime.utcnow)


# ─── Engine & Session ────────────────────────────────────────────────────────

engine = create_async_engine(settings.database_url, echo=False)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create all tables (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:  # type: ignore[misc]
    async with async_session_factory() as session:
        yield session
