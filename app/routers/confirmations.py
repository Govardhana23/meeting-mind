"""
Confirmations API router — the critical human-in-the-loop approval workflow.

Every extracted action item and detected conflict is queued as a PendingAction
that must be explicitly approved or rejected before any external side-effects
(task creation, notifications) are triggered.
"""

from __future__ import annotations

import json
import logging
import datetime as _dt

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import (
    PendingAction, ActionItem, Conflict,
    async_session_factory, get_session,
)
from app.models import PendingActionOut, PendingActionReview

logger = logging.getLogger("meeting_mind.confirmations")

router = APIRouter(prefix="/api/confirmations", tags=["confirmations"])


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _enrich_pending_action(pa: PendingAction) -> dict:
    """Convert a PendingAction ORM row to a dict with parsed payload."""
    data = {
        "id": pa.id,
        "meeting_id": pa.meeting_id,
        "action_type": pa.action_type,
        "reference_id": pa.reference_id,
        "status": pa.status,
        "created_at": pa.created_at.isoformat() if pa.created_at else None,
        "payload": None,
    }
    if pa.payload:
        try:
            data["payload"] = json.loads(pa.payload)
        except (json.JSONDecodeError, TypeError):
            data["payload"] = pa.payload
    return data


async def _execute_approval(
    session: AsyncSession,
    pa: PendingAction,
    edits: dict | None = None,
) -> None:
    """Carry out the side-effect for an approved PendingAction."""

    if pa.action_type == "create_task":
        # Apply optional edits to the ActionItem first
        action_item = await session.get(ActionItem, pa.reference_id)
        if action_item:
            if edits:
                if "description" in edits:
                    action_item.description = edits["description"]
                if "owner_name" in edits:
                    action_item.owner_name = edits["owner_name"]
                if "deadline" in edits and edits["deadline"]:
                    try:
                        action_item.deadline = _dt.datetime.fromisoformat(
                            edits["deadline"]
                        )
                    except (ValueError, TypeError):
                        pass
                if "priority" in edits:
                    action_item.priority = edits["priority"]

            # Attempt to push to the external task tracker
            try:
                from app.services.task_tracker import create_task_for_action_item
                result = await create_task_for_action_item(action_item)
                if result:
                    action_item.tracker_card_id = result.get("card_id")
                    action_item.tracker_card_url = result.get("card_url")
            except Exception:
                logger.warning(
                    "Task tracker unavailable for action %s — item confirmed locally.",
                    action_item.id,
                    exc_info=True,
                )

            action_item.status = "confirmed"

        pa.status = "executed"
        pa.reviewed_at = _dt.datetime.utcnow()

    elif pa.action_type == "notify_conflict":
        conflict = await session.get(Conflict, pa.reference_id)
        if conflict:
            conflict.status = "acknowledged" if conflict.status == "unresolved" else conflict.status

        pa.status = "executed"
        pa.reviewed_at = _dt.datetime.utcnow()

    elif pa.action_type == "send_reminder":
        try:
            from app.services.notifier import EmailNotifier
            notifier = EmailNotifier()

            payload = {}
            if pa.payload:
                try:
                    payload = json.loads(pa.payload)
                except (json.JSONDecodeError, TypeError):
                    pass

            await notifier.send_reminder(
                to_email=payload.get("email", ""),
                subject=payload.get("subject", "Reminder"),
                body=payload.get("body", ""),
            )
        except Exception:
            logger.warning(
                "Could not send reminder for pending action %s",
                pa.id,
                exc_info=True,
            )

        pa.status = "executed"
        pa.reviewed_at = _dt.datetime.utcnow()


async def _execute_rejection(session: AsyncSession, pa: PendingAction) -> None:
    """Handle rejection side-effects."""
    if pa.action_type == "create_task":
        action_item = await session.get(ActionItem, pa.reference_id)
        if action_item:
            action_item.status = "cancelled"

    pa.status = "rejected"
    pa.reviewed_at = _dt.datetime.utcnow()


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/", response_model=list[dict])
async def list_pending(session: AsyncSession = Depends(get_session)):
    """Return all pending actions awaiting review."""
    result = await session.execute(
        select(PendingAction)
        .where(PendingAction.status == "pending")
        .order_by(PendingAction.created_at)
    )
    rows = result.scalars().all()
    return [_enrich_pending_action(pa) for pa in rows]


@router.get("/meeting/{meeting_id}", response_model=list[dict])
async def list_pending_for_meeting(
    meeting_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Return pending actions for a specific meeting."""
    result = await session.execute(
        select(PendingAction)
        .where(
            PendingAction.meeting_id == meeting_id,
            PendingAction.status == "pending",
        )
        .order_by(PendingAction.created_at)
    )
    rows = result.scalars().all()
    return [_enrich_pending_action(pa) for pa in rows]


@router.post("/{action_id}/review")
async def review_action(
    action_id: str,
    body: PendingActionReview,
    session: AsyncSession = Depends(get_session),
):
    """Approve or reject a single pending action."""
    pa = await session.get(PendingAction, action_id)
    if not pa:
        raise HTTPException(status_code=404, detail="Pending action not found")
    if pa.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Action already reviewed (status={pa.status})",
        )

    if body.status == "approved":
        await _execute_approval(session, pa, edits=body.edits)
    else:
        await _execute_rejection(session, pa)

    await session.commit()
    return {"id": pa.id, "status": pa.status}


@router.post("/meeting/{meeting_id}/approve-all")
async def approve_all(
    meeting_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Bulk-approve every pending action for a meeting."""
    result = await session.execute(
        select(PendingAction)
        .where(
            PendingAction.meeting_id == meeting_id,
            PendingAction.status == "pending",
        )
        .order_by(PendingAction.created_at)
    )
    rows = result.scalars().all()

    processed = 0
    for pa in rows:
        await _execute_approval(session, pa)
        processed += 1

    await session.commit()
    return {"meeting_id": meeting_id, "approved": processed}


@router.post("/meeting/{meeting_id}/reject-all")
async def reject_all(
    meeting_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Bulk-reject every pending action for a meeting."""
    result = await session.execute(
        select(PendingAction)
        .where(
            PendingAction.meeting_id == meeting_id,
            PendingAction.status == "pending",
        )
        .order_by(PendingAction.created_at)
    )
    rows = result.scalars().all()

    processed = 0
    for pa in rows:
        await _execute_rejection(session, pa)
        processed += 1

    await session.commit()
    return {"meeting_id": meeting_id, "rejected": processed}
