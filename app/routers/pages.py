"""
Pages router — serves Jinja2 HTML templates for the web UI.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import (
    Meeting, Decision, ActionItem, Conflict, PendingAction,
    TeamMember, DigestLog,
    get_session,
)

router = APIRouter(tags=["pages"])

templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent.parent / "templates")
)


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Main dashboard page."""
    total_meetings = (await session.execute(select(func.count(Meeting.id)))).scalar_one()
    total_decisions = (await session.execute(select(func.count(Decision.id)))).scalar_one()
    total_action_items = (await session.execute(select(func.count(ActionItem.id)))).scalar_one()
    pending_confirmations = (
        await session.execute(
            select(func.count(PendingAction.id)).where(PendingAction.status == "pending")
        )
    ).scalar_one()
    unresolved_conflicts = (
        await session.execute(
            select(func.count(Conflict.id)).where(Conflict.status == "unresolved")
        )
    ).scalar_one()
    overdue_tasks = (
        await session.execute(
            select(func.count(ActionItem.id)).where(
                ActionItem.deadline < _dt.datetime.utcnow(),
                ActionItem.status.notin_(["done", "cancelled"]),
            )
        )
    ).scalar_one()

    recent_meetings_result = await session.execute(
        select(Meeting).order_by(Meeting.meeting_date.desc()).limit(10)
    )
    recent_meetings = recent_meetings_result.scalars().all()

    recent_conflicts_result = await session.execute(
        select(Conflict).order_by(Conflict.created_at.desc()).limit(5)
    )
    recent_conflicts = recent_conflicts_result.scalars().all()

    # Enrich conflicts
    enriched_conflicts = []
    for c in recent_conflicts:
        new_dec = await session.get(Decision, c.new_decision_id)
        old_dec = await session.get(Decision, c.old_decision_id)
        enriched_conflicts.append({
            "id": c.id,
            "severity": c.severity,
            "status": c.status,
            "explanation": c.explanation,
            "new_decision_text": new_dec.text if new_dec else "",
            "old_decision_text": old_dec.text if old_dec else "",
            "created_at": c.created_at,
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": {
            "total_meetings": total_meetings,
            "total_decisions": total_decisions,
            "total_action_items": total_action_items,
            "pending_confirmations": pending_confirmations,
            "unresolved_conflicts": unresolved_conflicts,
            "overdue_tasks": overdue_tasks,
        },
        "recent_meetings": recent_meetings,
        "recent_conflicts": enriched_conflicts,
    })


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Meeting upload page."""
    return templates.TemplateResponse("upload.html", {"request": request})


@router.get("/meetings/{meeting_id}", response_class=HTMLResponse)
async def meeting_detail_page(
    meeting_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Meeting detail page with transcript, decisions, actions, conflicts."""
    meeting = await session.get(Meeting, meeting_id)
    if not meeting:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "stats": {},
            "recent_meetings": [],
            "recent_conflicts": [],
        })

    # Gather conflicts related to this meeting's decisions
    decision_ids = [d.id for d in meeting.decisions]
    conflicts = []
    if decision_ids:
        result = await session.execute(
            select(Conflict).where(
                (Conflict.new_decision_id.in_(decision_ids))
                | (Conflict.old_decision_id.in_(decision_ids))
            )
        )
        conflict_rows = result.scalars().all()
        for c in conflict_rows:
            new_dec = await session.get(Decision, c.new_decision_id)
            old_dec = await session.get(Decision, c.old_decision_id)
            conflicts.append({
                "id": c.id,
                "severity": c.severity,
                "status": c.status,
                "explanation": c.explanation,
                "resolution": c.resolution,
                "new_decision_text": new_dec.text if new_dec else "",
                "old_decision_text": old_dec.text if old_dec else "",
                "new_decided_by": new_dec.decided_by if new_dec else "",
                "old_decided_by": old_dec.decided_by if old_dec else "",
                "created_at": c.created_at,
            })

    return templates.TemplateResponse("meeting_detail.html", {
        "request": request,
        "meeting": meeting,
        "utterances": sorted(meeting.utterances, key=lambda u: u.sequence),
        "decisions": meeting.decisions,
        "action_items": meeting.action_items,
        "conflicts": conflicts,
    })


@router.get("/confirmations", response_class=HTMLResponse)
async def confirmations_page(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Confirmation queue page — the critical human-in-the-loop step."""
    import json

    result = await session.execute(
        select(PendingAction)
        .where(PendingAction.status == "pending")
        .order_by(PendingAction.created_at)
    )
    pending_actions = result.scalars().all()

    # Group by meeting
    grouped: dict[str, dict] = {}
    for pa in pending_actions:
        if pa.meeting_id not in grouped:
            meeting = await session.get(Meeting, pa.meeting_id)
            grouped[pa.meeting_id] = {
                "meeting_id": pa.meeting_id,
                "meeting_title": meeting.title if meeting else "Unknown",
                "actions": [],
            }

        payload = {}
        if pa.payload:
            try:
                payload = json.loads(pa.payload)
            except (json.JSONDecodeError, TypeError):
                pass

        # Enrich based on type
        enriched = {
            "id": pa.id,
            "action_type": pa.action_type,
            "reference_id": pa.reference_id,
            "payload": payload,
            "created_at": pa.created_at,
        }

        if pa.action_type == "create_task":
            action_item = await session.get(ActionItem, pa.reference_id)
            if action_item:
                enriched["action_item"] = {
                    "id": action_item.id,
                    "description": action_item.description,
                    "owner_name": action_item.owner_name,
                    "deadline": action_item.deadline.isoformat() if action_item.deadline else "",
                    "priority": action_item.priority,
                }
        elif pa.action_type == "notify_conflict":
            conflict = await session.get(Conflict, pa.reference_id)
            if conflict:
                new_dec = await session.get(Decision, conflict.new_decision_id)
                old_dec = await session.get(Decision, conflict.old_decision_id)
                enriched["conflict"] = {
                    "id": conflict.id,
                    "explanation": conflict.explanation,
                    "severity": conflict.severity,
                    "new_decision_text": new_dec.text if new_dec else "",
                    "old_decision_text": old_dec.text if old_dec else "",
                }

        grouped[pa.meeting_id]["actions"].append(enriched)

    return templates.TemplateResponse("confirmations.html", {
        "request": request,
        "grouped_actions": list(grouped.values()),
        "total_pending": len(pending_actions),
    })


@router.get("/conflicts", response_class=HTMLResponse)
async def conflicts_page(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Conflicts view page."""
    result = await session.execute(
        select(Conflict).order_by(Conflict.created_at.desc())
    )
    rows = result.scalars().all()

    conflicts = []
    for c in rows:
        new_dec = await session.get(Decision, c.new_decision_id)
        old_dec = await session.get(Decision, c.old_decision_id)
        new_meeting = await session.get(Meeting, new_dec.meeting_id) if new_dec else None
        old_meeting = await session.get(Meeting, old_dec.meeting_id) if old_dec else None

        conflicts.append({
            "id": c.id,
            "severity": c.severity,
            "status": c.status,
            "explanation": c.explanation,
            "resolution": c.resolution,
            "new_decision_text": new_dec.text if new_dec else "",
            "old_decision_text": old_dec.text if old_dec else "",
            "new_meeting_title": new_meeting.title if new_meeting else "",
            "old_meeting_title": old_meeting.title if old_meeting else "",
            "new_decided_by": new_dec.decided_by if new_dec else "",
            "old_decided_by": old_dec.decided_by if old_dec else "",
            "created_at": c.created_at,
        })

    return templates.TemplateResponse("conflicts.html", {
        "request": request,
        "conflicts": conflicts,
    })


@router.get("/team", response_class=HTMLResponse)
async def team_page(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Team management page."""
    result = await session.execute(
        select(TeamMember).order_by(TeamMember.name)
    )
    members = result.scalars().all()

    # Count action items per member
    member_data = []
    for m in members:
        names = [m.name.lower()]
        if m.aliases:
            names.extend(a.strip().lower() for a in m.aliases.split(",") if a.strip())

        ai_count = 0
        for name in names:
            count_result = await session.execute(
                select(func.count(ActionItem.id)).where(
                    func.lower(ActionItem.owner_name) == name
                )
            )
            ai_count += count_result.scalar_one()

        member_data.append({
            "id": m.id,
            "name": m.name,
            "email": m.email,
            "aliases": m.aliases,
            "action_items_count": ai_count,
        })

    return templates.TemplateResponse("team.html", {
        "request": request,
        "members": member_data,
    })


@router.get("/digests", response_class=HTMLResponse)
async def digests_page(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Digest preview and sending page."""
    # Load members for the dropdown
    members_result = await session.execute(
        select(TeamMember).order_by(TeamMember.name)
    )
    members = members_result.scalars().all()

    # Load digest history
    history_result = await session.execute(
        select(DigestLog).order_by(DigestLog.created_at.desc()).limit(20)
    )
    history_rows = history_result.scalars().all()

    history = []
    for dl in history_rows:
        member = await session.get(TeamMember, dl.member_id)
        history.append({
            "id": dl.id,
            "member_name": member.name if member else "Unknown",
            "member_email": member.email if member else "",
            "week_start": dl.week_start.isoformat() if dl.week_start else "",
            "sent_at": dl.sent_at.strftime("%Y-%m-%d %H:%M") if dl.sent_at else "Not sent",
            "created_at": dl.created_at.strftime("%Y-%m-%d %H:%M") if dl.created_at else "",
        })

    return templates.TemplateResponse("digests.html", {
        "request": request,
        "members": members,
        "history": history,
    })
