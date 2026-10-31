"""
Cross-meeting decision memory store.

Maintains long-term decision persistence, vector/embedding computation,
retrieval of active and past decisions across meetings, and lifecycle status
(active, superseded, reverted).
"""

from __future__ import annotations

import json
import logging
import datetime as _dt
from typing import Optional, Sequence

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Decision, Meeting, Conflict

logger = logging.getLogger("meeting_mind.memory")


async def store_decisions(
    session: AsyncSession,
    decisions: Sequence[Decision] | list[dict],
    meeting_id: Optional[str] = None,
) -> list[Decision]:
    """
    Persist new decisions and generate their embeddings/semantic signatures.
    Accepts either Decision ORM objects or raw dicts.
    """
    from app.services.conflict_detector import compute_embedding

    persisted: list[Decision] = []

    for item in decisions:
        if isinstance(item, Decision):
            dec_obj = item
        else:
            dec_obj = Decision(
                meeting_id=meeting_id or item.get("meeting_id", ""),
                text=item.get("text", ""),
                context=item.get("context"),
                topic=item.get("topic"),
                decided_by=item.get("decided_by"),
                status=item.get("status", "active"),
            )
            session.add(dec_obj)

        # Compute embedding if not already set
        if not dec_obj.embedding_json and dec_obj.text:
            try:
                emb = await compute_embedding(dec_obj.text)
                if emb:
                    dec_obj.embedding_json = json.dumps(emb)
            except Exception as e:
                logger.warning("Could not compute embedding for decision '%s': %s", dec_obj.text[:30], e)

        persisted.append(dec_obj)

    await session.flush()
    return persisted


async def get_all_active_decisions(
    session: AsyncSession,
    exclude_meeting_id: Optional[str] = None,
) -> list[Decision]:
    """
    Retrieve all historical active decisions, optionally excluding a specific meeting.
    Eagerly loads meeting metadata for cross-meeting reasoning.
    """
    query = (
        select(Decision)
        .options(selectinload(Decision.meeting))
        .where(Decision.status == "active")
        .order_by(Decision.created_at.desc())
    )
    if exclude_meeting_id:
        query = query.where(Decision.meeting_id != exclude_meeting_id)

    result = await session.execute(query)
    return list(result.scalars().all())


async def get_decisions_by_topic(
    topic: str,
    session: AsyncSession,
) -> list[Decision]:
    """Find decisions matching a specific topic substring or keyword."""
    query = (
        select(Decision)
        .options(selectinload(Decision.meeting))
        .where(Decision.topic.ilike(f"%{topic}%"))
        .order_by(Decision.created_at.desc())
    )
    result = await session.execute(query)
    return list(result.scalars().all())


async def supersede_decision(
    decision_id: str,
    session: AsyncSession,
) -> Optional[Decision]:
    """Mark a decision as superseded by a newer decision."""
    decision = await session.get(Decision, decision_id)
    if decision:
        decision.status = "superseded"
        await session.commit()
    return decision


async def get_decision_history(
    session: AsyncSession,
    topic: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """
    Return a structured history of decisions across all meetings,
    including linked meeting titles, dates, statuses, and conflict links.
    """
    query = select(Decision).options(selectinload(Decision.meeting)).order_by(Decision.created_at.desc())
    if topic:
        query = query.where(Decision.topic.ilike(f"%{topic}%"))
    query = query.limit(limit)

    result = await session.execute(query)
    decisions = result.scalars().all()

    history = []
    for d in decisions:
        meeting_title = d.meeting.title if d.meeting else "Unknown Meeting"
        meeting_date = d.meeting.meeting_date.isoformat() if d.meeting and d.meeting.meeting_date else None
        history.append({
            "id": d.id,
            "meeting_id": d.meeting_id,
            "meeting_title": meeting_title,
            "meeting_date": meeting_date,
            "text": d.text,
            "topic": d.topic,
            "decided_by": d.decided_by,
            "status": d.status,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        })
    return history
