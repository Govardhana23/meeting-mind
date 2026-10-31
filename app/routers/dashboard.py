"""
Dashboard API router — stats, conflicts, team management, action items, digests.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import (
    Meeting, Decision, ActionItem, Conflict, PendingAction,
    TeamMember, DigestLog,
    async_session_factory, get_session,
)
from app.models import (
    ConflictOut, ConflictResolve,
    ActionItemOut, ActionItemUpdate,
    TeamMemberCreate, TeamMemberOut,
    DigestOut, DecisionOut,
)

logger = logging.getLogger("meeting_mind.dashboard")

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# ─── Stats ───────────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_stats(session: AsyncSession = Depends(get_session)):
    """Aggregate counts for the dashboard header cards."""

    total_meetings = (
        await session.execute(select(func.count(Meeting.id)))
    ).scalar_one()

    total_decisions = (
        await session.execute(select(func.count(Decision.id)))
    ).scalar_one()

    total_action_items = (
        await session.execute(select(func.count(ActionItem.id)))
    ).scalar_one()

    pending_confirmations = (
        await session.execute(
            select(func.count(PendingAction.id))
            .where(PendingAction.status == "pending")
        )
    ).scalar_one()

    unresolved_conflicts = (
        await session.execute(
            select(func.count(Conflict.id))
            .where(Conflict.status == "unresolved")
        )
    ).scalar_one()

    overdue_tasks = (
        await session.execute(
            select(func.count(ActionItem.id))
            .where(
                ActionItem.deadline < _dt.datetime.utcnow(),
                ActionItem.status.notin_(["done", "cancelled"]),
            )
        )
    ).scalar_one()

    return {
        "total_meetings": total_meetings,
        "total_decisions": total_decisions,
        "total_action_items": total_action_items,
        "pending_confirmations": pending_confirmations,
        "unresolved_conflicts": unresolved_conflicts,
        "overdue_tasks": overdue_tasks,
    }


# ─── Conflicts ───────────────────────────────────────────────────────────────

@router.get("/conflicts", response_model=list[ConflictOut])
async def list_conflicts(
    status: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    """List conflicts with enriched decision/meeting info."""
    stmt = select(Conflict).order_by(Conflict.created_at.desc())
    if status:
        stmt = stmt.where(Conflict.status == status)

    result = await session.execute(stmt)
    rows = result.scalars().all()

    enriched: list[ConflictOut] = []
    for c in rows:
        new_dec = await session.get(Decision, c.new_decision_id)
        old_dec = await session.get(Decision, c.old_decision_id)

        new_meeting = await session.get(Meeting, new_dec.meeting_id) if new_dec else None
        old_meeting = await session.get(Meeting, old_dec.meeting_id) if old_dec else None

        enriched.append(ConflictOut(
            id=c.id,
            new_decision_id=c.new_decision_id,
            old_decision_id=c.old_decision_id,
            explanation=c.explanation,
            severity=c.severity,
            resolution=c.resolution,
            status=c.status,
            created_at=c.created_at,
            new_decision_text=new_dec.text if new_dec else None,
            old_decision_text=old_dec.text if old_dec else None,
            new_meeting_title=new_meeting.title if new_meeting else None,
            old_meeting_title=old_meeting.title if old_meeting else None,
        ))

    return enriched


@router.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict(
    conflict_id: str,
    body: ConflictResolve,
    session: AsyncSession = Depends(get_session),
):
    """Resolve a conflict and optionally supersede one of the decisions."""
    conflict = await session.get(Conflict, conflict_id)
    if not conflict:
        raise HTTPException(status_code=404, detail="Conflict not found")

    conflict.resolution = body.resolution
    conflict.status = "resolved"

    if body.keep_decision == "new":
        old_dec = await session.get(Decision, conflict.old_decision_id)
        if old_dec:
            old_dec.status = "superseded"
    elif body.keep_decision == "old":
        new_dec = await session.get(Decision, conflict.new_decision_id)
        if new_dec:
            new_dec.status = "superseded"
    # "both" → keep both active, no status change

    await session.commit()
    return {"id": conflict.id, "status": conflict.status, "resolution": conflict.resolution}


# ─── Team ────────────────────────────────────────────────────────────────────

@router.get("/team", response_model=list[TeamMemberOut])
async def list_team(session: AsyncSession = Depends(get_session)):
    """List all team members."""
    result = await session.execute(
        select(TeamMember).order_by(TeamMember.name)
    )
    return result.scalars().all()


@router.post("/team", response_model=TeamMemberOut, status_code=201)
async def create_team_member(
    body: TeamMemberCreate,
    session: AsyncSession = Depends(get_session),
):
    """Register a new team member."""
    member = TeamMember(
        name=body.name,
        email=body.email,
        aliases=body.aliases,
    )
    session.add(member)
    await session.commit()
    await session.refresh(member)
    return member


@router.delete("/team/{member_id}")
async def delete_team_member(
    member_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Remove a team member."""
    member = await session.get(TeamMember, member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Team member not found")
    await session.delete(member)
    await session.commit()
    return {"deleted": member_id}


# ─── Digest ──────────────────────────────────────────────────────────────────

@router.get("/digest/preview/{member_id}")
async def preview_digest(
    member_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Build a digest preview for a specific team member without sending."""
    member = await session.get(TeamMember, member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Team member not found")

    now = _dt.datetime.utcnow()
    week_start = now - _dt.timedelta(days=now.weekday(), hours=now.hour, minutes=now.minute, seconds=now.second)

    # Action items owned by this member
    member_names = [member.name.lower()]
    if member.aliases:
        member_names.extend(
            a.strip().lower() for a in member.aliases.split(",") if a.strip()
        )

    result = await session.execute(select(ActionItem))
    all_items = result.scalars().all()
    member_items = [
        ai for ai in all_items
        if ai.owner_name.lower() in member_names
    ]

    # Decisions from meetings containing these action items
    meeting_ids = {ai.meeting_id for ai in member_items}
    decisions = []
    if meeting_ids:
        dec_result = await session.execute(
            select(Decision).where(Decision.meeting_id.in_(meeting_ids))
        )
        decisions = dec_result.scalars().all()

    # Conflicts involving those decisions
    decision_ids = {d.id for d in decisions}
    conflicts = []
    if decision_ids:
        conf_result = await session.execute(
            select(Conflict).where(
                (Conflict.new_decision_id.in_(decision_ids))
                | (Conflict.old_decision_id.in_(decision_ids))
            )
        )
        conflicts = conf_result.scalars().all()

    # Build summary
    pending_count = sum(1 for ai in member_items if ai.status in ("pending", "confirmed"))
    done_count = sum(1 for ai in member_items if ai.status == "done")
    overdue = [
        ai for ai in member_items
        if ai.deadline and ai.deadline < now and ai.status not in ("done", "cancelled")
    ]

    summary_lines = [
        f"Weekly digest for {member.name}",
        f"Period: {week_start.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}",
        f"",
        f"Action Items: {len(member_items)} total, {done_count} done, {pending_count} active",
    ]
    if overdue:
        summary_lines.append(f"⚠️ {len(overdue)} overdue item(s)")
    if conflicts:
        summary_lines.append(f"⚠️ {len(conflicts)} conflict(s) involve your decisions")

    return DigestOut(
        member_name=member.name,
        member_email=member.email,
        week_start=week_start.date(),
        action_items=[ActionItemOut.model_validate(ai) for ai in member_items],
        decisions_involved=[DecisionOut.model_validate(d) for d in decisions],
        conflicts=[
            ConflictOut(
                id=c.id,
                new_decision_id=c.new_decision_id,
                old_decision_id=c.old_decision_id,
                explanation=c.explanation,
                severity=c.severity,
                resolution=c.resolution,
                status=c.status,
                created_at=c.created_at,
            )
            for c in conflicts
        ],
        summary="\n".join(summary_lines),
    )


@router.post("/digest/send")
async def send_digests(session: AsyncSession = Depends(get_session)):
    """Generate and send weekly digests to all team members."""
    result = await session.execute(select(TeamMember))
    members = result.scalars().all()

    sent = 0
    errors = 0
    for member in members:
        try:
            digest_data = await preview_digest(member.id, session)

            now = _dt.datetime.utcnow()
            week_start = now - _dt.timedelta(days=now.weekday())

            log_entry = DigestLog(
                member_id=member.id,
                week_start=week_start.date(),
                content=digest_data.summary,
                sent_at=now,
            )
            session.add(log_entry)

            try:
                from app.services.notifier import EmailNotifier
                notifier = EmailNotifier()
                await notifier.send_digest(
                    to_email=member.email,
                    subject=f"Meeting Mind Weekly Digest — {member.name}",
                    body=digest_data.summary,
                )
            except Exception:
                logger.warning("Email send failed for %s", member.email, exc_info=True)

            sent += 1
        except Exception:
            logger.error("Digest generation failed for %s", member.name, exc_info=True)
            errors += 1

    await session.commit()
    return {"sent": sent, "errors": errors, "total_members": len(members)}


@router.get("/digest/history")
async def digest_history(session: AsyncSession = Depends(get_session)):
    """List past digests."""
    result = await session.execute(
        select(DigestLog).order_by(DigestLog.created_at.desc())
    )
    rows = result.scalars().all()

    enriched = []
    for dl in rows:
        member = await session.get(TeamMember, dl.member_id)
        enriched.append({
            "id": dl.id,
            "member_name": member.name if member else "Unknown",
            "member_email": member.email if member else "",
            "week_start": dl.week_start.isoformat() if dl.week_start else None,
            "sent_at": dl.sent_at.isoformat() if dl.sent_at else None,
            "created_at": dl.created_at.isoformat() if dl.created_at else None,
        })

    return enriched


# ─── Action Items ────────────────────────────────────────────────────────────

@router.get("/action-items", response_model=list[ActionItemOut])
async def list_action_items(
    status: Optional[str] = Query(None),
    owner: Optional[str] = Query(None),
    overdue: Optional[bool] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    """List action items with optional filters."""
    stmt = select(ActionItem).order_by(ActionItem.created_at.desc())

    if status:
        stmt = stmt.where(ActionItem.status == status)
    if owner:
        stmt = stmt.where(ActionItem.owner_name.ilike(f"%{owner}%"))
    if overdue:
        stmt = stmt.where(
            ActionItem.deadline < _dt.datetime.utcnow(),
            ActionItem.status.notin_(["done", "cancelled"]),
        )

    result = await session.execute(stmt)
    return result.scalars().all()


@router.patch("/action-items/{item_id}", response_model=ActionItemOut)
async def update_action_item(
    item_id: str,
    body: ActionItemUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Partially update an action item."""
    item = await session.get(ActionItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Action item not found")

    if body.description is not None:
        item.description = body.description
    if body.owner_name is not None:
        item.owner_name = body.owner_name
    if body.deadline is not None:
        item.deadline = body.deadline
    if body.priority is not None:
        item.priority = body.priority
    if body.status is not None:
        item.status = body.status

    await session.commit()
    await session.refresh(item)
    return item
