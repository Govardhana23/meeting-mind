"""
Background scheduler that checks for upcoming deadlines and triggers
reminder emails via the confirmation queue.

Uses APScheduler ``BackgroundScheduler`` with ``asyncio.run()`` bridging
to keep the async DB/email code intact.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings
from app.database import (
    ActionItem,
    Meeting,
    PendingAction,
    ReminderLog,
    TeamMember,
)
from app.services.notifier import EmailNotifier

logger = logging.getLogger(__name__)


class ReminderScheduler:
    """Checks upcoming deadlines and orchestrates reminder dispatch."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        notifier: EmailNotifier,
    ) -> None:
        self._session_factory = session_factory
        self._notifier = notifier

    # ── deadline scan ─────────────────────────────────────────────────────

    async def check_upcoming_deadlines(self) -> list[dict]:
        """Scan active action-items and queue reminders for approaching deadlines.

        For each configured ``reminder_hours_before`` interval the method:
        1. Checks whether the deadline falls within that window.
        2. Ensures a reminder for this interval hasn't already been sent
           (``reminders_log``) or queued (``pending_actions``).
        3. Creates a ``PendingAction(type='send_reminder')`` for human
           confirmation before actual dispatch.

        Returns a list of dicts describing the pending reminders created.
        """
        created: list[dict] = []
        now = _dt.datetime.utcnow()

        async with self._session_factory() as session:
            # Fetch action items with upcoming deadlines
            result = await session.execute(
                select(ActionItem).where(
                    and_(
                        ActionItem.status.in_(["confirmed", "in_progress"]),
                        ActionItem.deadline.isnot(None),
                        ActionItem.deadline > now,
                    )
                )
            )
            items: list[ActionItem] = list(result.scalars().all())

            for item in items:
                for hours_before in settings.reminder_hours_before:
                    deadline: _dt.datetime = item.deadline  # type: ignore[assignment]
                    window_start = deadline - _dt.timedelta(hours=hours_before)

                    # Only queue if we're inside the window (now >= window_start)
                    if now < window_start:
                        continue

                    # Already sent?
                    already_sent = await session.execute(
                        select(ReminderLog).where(
                            and_(
                                ReminderLog.action_item_id == item.id,
                                ReminderLog.hours_before == hours_before,
                            )
                        )
                    )
                    if already_sent.scalar_one_or_none() is not None:
                        continue

                    # Already queued as pending?
                    already_queued = await session.execute(
                        select(PendingAction).where(
                            and_(
                                PendingAction.action_type == "send_reminder",
                                PendingAction.reference_id == item.id,
                                PendingAction.status.in_(["pending", "approved"]),
                            )
                        )
                    )
                    existing_pending = already_queued.scalars().all()
                    # Check if this specific hours_before is already queued
                    skip = False
                    for pa in existing_pending:
                        try:
                            payload = json.loads(pa.payload or "{}")
                            if payload.get("hours_before") == hours_before:
                                skip = True
                                break
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if skip:
                        continue

                    # Look up meeting title
                    meeting_title = ""
                    if item.meeting_id:
                        mtg_result = await session.execute(
                            select(Meeting.title).where(Meeting.id == item.meeting_id)
                        )
                        row = mtg_result.first()
                        if row:
                            meeting_title = row[0]

                    # Resolve owner email
                    owner_email = ""
                    if item.owner_id:
                        member_result = await session.execute(
                            select(TeamMember.email).where(TeamMember.id == item.owner_id)
                        )
                        email_row = member_result.first()
                        if email_row:
                            owner_email = email_row[0]

                    hours_remaining = max(1, int((deadline - now).total_seconds() / 3600))

                    payload_dict: dict[str, Any] = {
                        "action_item_id": item.id,
                        "action_item_desc": item.description,
                        "owner_email": owner_email,
                        "owner_name": item.owner_name,
                        "deadline": deadline.isoformat(),
                        "meeting_title": meeting_title,
                        "hours_before": hours_before,
                        "hours_remaining": hours_remaining,
                    }

                    pending = PendingAction(
                        meeting_id=item.meeting_id,
                        action_type="send_reminder",
                        reference_id=item.id,
                        payload=json.dumps(payload_dict),
                        status="pending",
                    )
                    session.add(pending)
                    created.append(payload_dict)

            if created:
                await session.commit()
                logger.info("Queued %d reminder(s) for confirmation.", len(created))

        return created

    # ── execute approved ──────────────────────────────────────────────────

    async def execute_approved_reminders(self) -> int:
        """Send emails for all approved ``send_reminder`` pending-actions.

        Returns the count of reminders successfully sent.
        """
        sent_count = 0

        async with self._session_factory() as session:
            result = await session.execute(
                select(PendingAction).where(
                    and_(
                        PendingAction.action_type == "send_reminder",
                        PendingAction.status == "approved",
                    )
                )
            )
            approved: list[PendingAction] = list(result.scalars().all())

            for pa in approved:
                try:
                    payload = json.loads(pa.payload or "{}")
                except (json.JSONDecodeError, TypeError):
                    logger.error("Invalid payload for pending action %s", pa.id)
                    pa.status = "executed"
                    pa.reviewed_at = _dt.datetime.utcnow()
                    continue

                to_email = payload.get("owner_email", "")
                if not to_email:
                    logger.warning(
                        "No email for reminder %s — marking executed.", pa.id
                    )
                    pa.status = "executed"
                    pa.reviewed_at = _dt.datetime.utcnow()
                    continue

                success = await self._notifier.send_reminder(
                    to_email=to_email,
                    action_item_desc=payload.get("action_item_desc", ""),
                    deadline=payload.get("deadline", ""),
                    meeting_title=payload.get("meeting_title", ""),
                    hours_remaining=payload.get("hours_remaining", 0),
                )

                if success:
                    # Log the sent reminder
                    log_entry = ReminderLog(
                        action_item_id=payload["action_item_id"],
                        hours_before=payload.get("hours_before", 0),
                        channel="email",
                    )
                    session.add(log_entry)
                    sent_count += 1

                pa.status = "executed"
                pa.reviewed_at = _dt.datetime.utcnow()

            if approved:
                await session.commit()

        if sent_count:
            logger.info("Sent %d reminder email(s).", sent_count)
        return sent_count

    # ── combined cycle ────────────────────────────────────────────────────

    async def run_check_cycle(self) -> None:
        """Run a full check-and-execute cycle (called by APScheduler)."""
        try:
            await self.check_upcoming_deadlines()
            await self.execute_approved_reminders()
        except Exception:
            logger.exception("Error during reminder check cycle.")


# ─── APScheduler setup ───────────────────────────────────────────────────────


def setup_scheduler(app_state: Any) -> BackgroundScheduler:
    """Create and start a ``BackgroundScheduler`` that runs the reminder
    check cycle every 30 minutes.
    """
    if hasattr(app_state, "session_factory"):
        session_factory: async_sessionmaker[AsyncSession] = app_state.session_factory
        notifier: EmailNotifier = getattr(app_state, "notifier", EmailNotifier())
    else:
        session_factory = app_state
        notifier = EmailNotifier()

    scheduler_instance = ReminderScheduler(session_factory, notifier)

    def _sync_job() -> None:
        """Bridge sync APScheduler callback to async code."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule into the existing loop (e.g. inside uvicorn)
                asyncio.ensure_future(scheduler_instance.run_check_cycle())
            else:
                asyncio.run(scheduler_instance.run_check_cycle())
        except RuntimeError:
            # No event loop exists yet — create one
            asyncio.run(scheduler_instance.run_check_cycle())

    bg_scheduler = BackgroundScheduler()
    bg_scheduler.add_job(
        _sync_job,
        "interval",
        minutes=30,
        id="reminder_check_cycle",
        replace_existing=True,
    )
    bg_scheduler.start()
    logger.info("Reminder scheduler started — checking every 30 minutes.")
    return bg_scheduler
