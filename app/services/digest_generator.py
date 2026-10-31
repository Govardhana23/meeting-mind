"""
Weekly digest generator.

Aggregates action-items, decisions, and conflicts per team member for
the past 7 days, then uses Gemini to produce a natural-language summary.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from google import genai
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import (
    ActionItem,
    Conflict,
    Decision,
    DigestLog,
    Meeting,
    TeamMember,
    async_session_factory,
)
from app.services.notifier import EmailNotifier

logger = logging.getLogger(__name__)


def _week_start_date() -> _dt.date:
    """Return the Monday of the current week (ISO calendar)."""
    today = _dt.date.today()
    return today - _dt.timedelta(days=today.weekday())


def _seven_days_ago() -> _dt.datetime:
    return _dt.datetime.utcnow() - _dt.timedelta(days=7)


# ─── Per-member digest ───────────────────────────────────────────────────────


async def generate_digest_for_member(
    member_id: str,
    session: AsyncSession,
) -> dict:
    """Build a structured digest for a single team member.

    Gathers:
    * Action-items owned by the member from the past 7 days.
    * Decisions from meetings in the past 7 days where the member was
      the owner of at least one action-item.
    * Unresolved conflicts related to those decisions.

    Uses Gemini to generate a natural-language HTML summary.

    Returns
    -------
    dict
        ``member_id``, ``member_name``, ``member_email``, ``week_start``,
        ``action_items``, ``decisions``, ``conflicts``, ``summary_html``.
    """
    cutoff = _seven_days_ago()
    week_start = _week_start_date()

    # Fetch the member
    member_result = await session.execute(
        select(TeamMember).where(TeamMember.id == member_id)
    )
    member: TeamMember | None = member_result.scalar_one_or_none()
    if member is None:
        return {
            "member_id": member_id,
            "member_name": "Unknown",
            "member_email": "",
            "week_start": week_start,
            "action_items": [],
            "decisions": [],
            "conflicts": [],
            "summary_html": "<p>Member not found.</p>",
        }

    # Action items from the past 7 days
    ai_result = await session.execute(
        select(ActionItem).where(
            and_(
                ActionItem.owner_id == member_id,
                ActionItem.created_at >= cutoff,
            )
        )
    )
    action_items: list[ActionItem] = list(ai_result.scalars().all())

    # Meeting IDs this member participated in (via action-items)
    member_meeting_ids: set[str] = {ai.meeting_id for ai in action_items if ai.meeting_id}

    # Also include meetings from the past 7 days where the member was an action-item owner
    all_ai_result = await session.execute(
        select(ActionItem.meeting_id).where(
            and_(
                ActionItem.owner_id == member_id,
                ActionItem.meeting_id.isnot(None),
            )
        )
    )
    for row in all_ai_result.all():
        member_meeting_ids.add(row[0])

    # Decisions from those meetings in the past 7 days
    decisions: list[Decision] = []
    if member_meeting_ids:
        dec_result = await session.execute(
            select(Decision).where(
                and_(
                    Decision.meeting_id.in_(list(member_meeting_ids)),
                    Decision.created_at >= cutoff,
                )
            )
        )
        decisions = list(dec_result.scalars().all())

    # Unresolved conflicts related to those decisions
    decision_ids = [d.id for d in decisions]
    conflicts: list[Conflict] = []
    if decision_ids:
        conf_result = await session.execute(
            select(Conflict).where(
                and_(
                    or_(
                        Conflict.new_decision_id.in_(decision_ids),
                        Conflict.old_decision_id.in_(decision_ids),
                    ),
                    Conflict.status == "unresolved",
                )
            )
        )
        conflicts = list(conf_result.scalars().all())

    # Build the Gemini prompt for summary generation
    summary_html = await _generate_summary(member, action_items, decisions, conflicts)

    return {
        "member_id": member_id,
        "member_name": member.name,
        "member_email": member.email,
        "week_start": week_start,
        "action_items": action_items,
        "decisions": decisions,
        "conflicts": conflicts,
        "summary_html": summary_html,
    }


async def _generate_summary(
    member: TeamMember,
    action_items: list[ActionItem],
    decisions: list[Decision],
    conflicts: list[Conflict],
) -> str:
    """Use Gemini to create a concise HTML digest summary."""
    if not settings.gemini_configured:
        logger.warning("Gemini not configured — falling back to basic summary.")
        return _fallback_summary(member, action_items, decisions, conflicts)

    # Prepare structured text for the LLM
    ai_lines = []
    for ai in action_items:
        deadline_str = ai.deadline.strftime("%Y-%m-%d") if ai.deadline else "No deadline"
        ai_lines.append(
            f"- [{ai.status.upper()}] {ai.description} (deadline: {deadline_str}, priority: {ai.priority})"
        )

    dec_lines = [f"- {d.text} (topic: {d.topic or 'N/A'}, status: {d.status})" for d in decisions]
    conf_lines = [f"- {c.explanation} (severity: {c.severity})" for c in conflicts]

    prompt = f"""\
You are a concise professional assistant.  Generate an HTML snippet (no full
page — just the content inside <div>) summarising this team member's week.

**Member:** {member.name}

**Action Items ({len(action_items)}):**
{chr(10).join(ai_lines) or "None this week."}

**Decisions Involved ({len(decisions)}):**
{chr(10).join(dec_lines) or "None this week."}

**Unresolved Conflicts ({len(conflicts)}):**
{chr(10).join(conf_lines) or "None."}

Requirements:
- Use headings (<h3>), bullet lists, and short paragraphs.
- Highlight overdue or urgent items in <strong> tags.
- Keep the tone professional but friendly.
- Maximum 250 words.
- Use inline CSS only (no classes or external styles).
- Output ONLY the HTML snippet, nothing else.
"""

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw_text = response.text or ""
        # Strip markdown code fences if the model wraps them
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_text = "\n".join(lines)
        return raw_text.strip()
    except Exception:
        logger.exception("Gemini digest generation failed — using fallback.")
        return _fallback_summary(member, action_items, decisions, conflicts)


def _fallback_summary(
    member: TeamMember,
    action_items: list[ActionItem],
    decisions: list[Decision],
    conflicts: list[Conflict],
) -> str:
    """Plain HTML summary when Gemini is unavailable."""
    parts = [f"<h3>Weekly Summary for {member.name}</h3>"]

    # Action items
    parts.append(f"<h4>Action Items ({len(action_items)})</h4>")
    if action_items:
        parts.append("<ul>")
        for ai in action_items:
            deadline_str = ai.deadline.strftime("%Y-%m-%d") if ai.deadline else "No deadline"
            parts.append(
                f"<li><strong>[{ai.status.upper()}]</strong> {ai.description} "
                f"— deadline: {deadline_str}, priority: {ai.priority}</li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p>No action items this week.</p>")

    # Decisions
    parts.append(f"<h4>Decisions ({len(decisions)})</h4>")
    if decisions:
        parts.append("<ul>")
        for d in decisions:
            parts.append(f"<li>{d.text} <em>(topic: {d.topic or 'N/A'})</em></li>")
        parts.append("</ul>")
    else:
        parts.append("<p>No decisions this week.</p>")

    # Conflicts
    if conflicts:
        parts.append(f"<h4>⚠️ Unresolved Conflicts ({len(conflicts)})</h4><ul>")
        for c in conflicts:
            parts.append(f"<li>{c.explanation} — severity: <strong>{c.severity}</strong></li>")
        parts.append("</ul>")

    return "\n".join(parts)


# ─── Bulk digest generation ─────────────────────────────────────────────────


async def generate_all_digests(session: AsyncSession) -> list[dict]:
    """Generate weekly digests for every team member and persist logs.

    Returns a list of digest dicts (one per member).
    """
    week_start = _week_start_date()

    result = await session.execute(select(TeamMember))
    members: list[TeamMember] = list(result.scalars().all())

    digests: list[dict] = []
    for member in members:
        digest = await generate_digest_for_member(member.id, session)

        # Persist a DigestLog entry
        log_entry = DigestLog(
            member_id=member.id,
            week_start=week_start,
            content=digest.get("summary_html", ""),
        )
        session.add(log_entry)
        digest["digest_log_id"] = log_entry.id
        digests.append(digest)

    await session.commit()
    logger.info("Generated %d weekly digest(s).", len(digests))
    return digests


# ─── Send all digests ────────────────────────────────────────────────────────


async def send_all_digests(
    session: AsyncSession,
    notifier: EmailNotifier,
) -> int:
    """Generate digests for all members and email them.

    Returns the count of emails successfully sent.
    """
    if not settings.smtp_configured:
        logger.warning("SMTP not configured — digest emails will not be sent.")
        return 0

    digests = await generate_all_digests(session)

    sent = 0
    for digest in digests:
        email = digest.get("member_email", "")
        if not email:
            continue

        success = await notifier.send_digest(
            to_email=email,
            member_name=digest.get("member_name", ""),
            digest_html=digest.get("summary_html", ""),
            week_start=str(digest.get("week_start", "")),
        )
        if success:
            # Mark the digest log as sent
            if "digest_log_id" in digest:
                log_result = await session.execute(
                    select(DigestLog).where(DigestLog.id == digest["digest_log_id"])
                )
                log_entry = log_result.scalar_one_or_none()
                if log_entry:
                    log_entry.sent_at = _dt.datetime.utcnow()
            sent += 1

    await session.commit()
    logger.info("Sent %d digest email(s).", sent)
    return sent
