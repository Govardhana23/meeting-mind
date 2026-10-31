"""
Meetings API router — upload, process, list, detail, and status polling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import datetime as _dt
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import (
    Meeting, Utterance, Decision, ActionItem, Conflict, PendingAction,
    async_session_factory, get_session,
)
from app.models import MeetingUpload, MeetingOut, MeetingDetail, ProcessingResult

logger = logging.getLogger("meeting_mind.meetings")

router = APIRouter(prefix="/api/meetings", tags=["meetings"])


# ─── Background processing ──────────────────────────────────────────────────

async def process_meeting(meeting_id: str) -> None:
    """
    Full pipeline: parse transcript → extract decisions/actions → detect
    conflicts → queue pending actions.  Runs as a background task.
    """
    async with async_session_factory() as session:
        try:
            # ── Load meeting ────────────────────────────────────────────
            meeting = await session.get(Meeting, meeting_id)
            if meeting is None:
                logger.error("process_meeting: meeting %s not found", meeting_id)
                return

            meeting.status = "processing"
            await session.commit()

            # ── 1. Parse transcript ─────────────────────────────────────
            from app.services.transcript_parser import parse_transcript

            parsed_utterances = parse_transcript(meeting.raw_transcript, meeting.format)

            for seq, u in enumerate(parsed_utterances):
                utterance = Utterance(
                    meeting_id=meeting_id,
                    speaker=u.get("speaker", "Unknown"),
                    text=u.get("text", ""),
                    start_time=u.get("start_time"),
                    end_time=u.get("end_time"),
                    sequence=seq,
                )
                session.add(utterance)

            await session.flush()

            # ── 2. Extract decisions & action items ─────────────────────
            from app.services.llm_extractor import extract_decisions_and_actions

            extraction = await extract_decisions_and_actions(
                meeting.raw_transcript, meeting.title
            )

            # Store decisions
            decision_rows: list[Decision] = []
            for d in extraction.get("decisions", []):
                decision = Decision(
                    meeting_id=meeting_id,
                    text=d.get("text", ""),
                    context=d.get("context"),
                    topic=d.get("topic"),
                    decided_by=d.get("decided_by"),
                    status="active",
                )
                session.add(decision)
                decision_rows.append(decision)

            await session.flush()

            # Persist into decision memory store
            from app.memory.decision_store import store_decisions

            await store_decisions(session, decision_rows)

            # Store action items
            action_rows: list[ActionItem] = []
            for a in extraction.get("action_items", []):
                # Try to link to the first matching decision by topic
                linked_decision_id = None
                if a.get("topic"):
                    for dr in decision_rows:
                        if dr.topic and a["topic"].lower() in dr.topic.lower():
                            linked_decision_id = dr.id
                            break

                deadline = None
                if a.get("deadline"):
                    try:
                        deadline = _dt.datetime.fromisoformat(a["deadline"])
                    except (ValueError, TypeError):
                        pass

                item = ActionItem(
                    meeting_id=meeting_id,
                    decision_id=linked_decision_id,
                    description=a.get("description", ""),
                    owner_name=a.get("owner", "Unassigned"),
                    deadline=deadline,
                    priority=a.get("priority", "medium"),
                    status="pending",
                )
                session.add(item)
                action_rows.append(item)

            await session.flush()

            # ── 3. Detect conflicts ─────────────────────────────────────
            from app.services.conflict_detector import detect_conflicts

            conflicts = await detect_conflicts(session, decision_rows)

            conflict_rows: list[Conflict] = []
            for c in conflicts:
                conflict = Conflict(
                    new_decision_id=c["new_decision_id"],
                    old_decision_id=c["old_decision_id"],
                    explanation=c.get("explanation", ""),
                    severity=c.get("severity", "medium"),
                    status="unresolved",
                )
                session.add(conflict)
                conflict_rows.append(conflict)

            await session.flush()

            # ── 4. Create pending actions for confirmation queue ────────
            for item in action_rows:
                pa = PendingAction(
                    meeting_id=meeting_id,
                    action_type="create_task",
                    reference_id=item.id,
                    payload=json.dumps({
                        "description": item.description,
                        "owner_name": item.owner_name,
                        "deadline": item.deadline.isoformat() if item.deadline else None,
                        "priority": item.priority,
                    }),
                    status="pending",
                )
                session.add(pa)

            for conflict in conflict_rows:
                pa = PendingAction(
                    meeting_id=meeting_id,
                    action_type="notify_conflict",
                    reference_id=conflict.id,
                    payload=json.dumps({
                        "explanation": conflict.explanation,
                        "severity": conflict.severity,
                        "new_decision_id": conflict.new_decision_id,
                        "old_decision_id": conflict.old_decision_id,
                    }),
                    status="pending",
                )
                session.add(pa)

            # ── 5. Mark complete ────────────────────────────────────────
            meeting.status = "processed"
            await session.commit()

            logger.info(
                "Meeting %s processed: %d utterances, %d decisions, %d actions, %d conflicts",
                meeting_id,
                len(parsed_utterances),
                len(decision_rows),
                len(action_rows),
                len(conflict_rows),
            )

        except Exception:
            logger.error(
                "Error processing meeting %s:\n%s",
                meeting_id, traceback.format_exc(),
            )
            try:
                meeting = await session.get(Meeting, meeting_id)
                if meeting:
                    meeting.status = "error"
                    await session.commit()
            except Exception:
                logger.error("Failed to mark meeting %s as error", meeting_id)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/upload", response_model=MeetingOut, status_code=201)
async def upload_meeting(
    body: MeetingUpload,
    session: AsyncSession = Depends(get_session),
):
    """Accept a meeting transcript and kick off background processing."""
    meeting = Meeting(
        title=body.title,
        raw_transcript=body.transcript,
        format=body.format,
        meeting_date=body.meeting_date or _dt.datetime.utcnow(),
        status="processing",
    )
    session.add(meeting)
    await session.commit()
    await session.refresh(meeting)

    # Fire-and-forget background task
    asyncio.create_task(process_meeting(meeting.id))

    return meeting


@router.get("/", response_model=list[MeetingOut])
async def list_meetings(session: AsyncSession = Depends(get_session)):
    """List all meetings, newest first."""
    result = await session.execute(
        select(Meeting).order_by(Meeting.meeting_date.desc())
    )
    return result.scalars().all()


@router.get("/{meeting_id}", response_model=MeetingDetail)
async def get_meeting(meeting_id: str, session: AsyncSession = Depends(get_session)):
    """Return full meeting detail with utterances, decisions, and action items."""
    meeting = await session.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Relationships are lazy="selectin" so they're already loaded.
    return meeting


@router.get("/{meeting_id}/status")
async def get_meeting_status(
    meeting_id: str, session: AsyncSession = Depends(get_session)
):
    """Lightweight status endpoint for polling during processing."""
    meeting = await session.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return {"meeting_id": meeting.id, "status": meeting.status}
