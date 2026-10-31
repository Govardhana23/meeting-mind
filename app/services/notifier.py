"""
Email notification service using aiosmtplib.

Provides ``EmailNotifier`` with methods for sending reminders, conflict
notifications, and weekly digests.  All emails use professional inline-CSS
HTML formatting.
"""

from __future__ import annotations

import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Shared HTML style fragments ─────────────────────────────────────────────

_BASE_STYLE = """
<style>
  body { margin: 0; padding: 0; background-color: #f4f6f9; font-family: 'Segoe UI', Arial, sans-serif; }
  .container { max-width: 600px; margin: 30px auto; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .header { background: linear-gradient(135deg, #4f46e5, #7c3aed); color: #ffffff; padding: 28px 32px; }
  .header h1 { margin: 0; font-size: 22px; font-weight: 600; }
  .header p { margin: 6px 0 0; font-size: 13px; opacity: 0.85; }
  .body { padding: 28px 32px; color: #1e293b; line-height: 1.65; }
  .body h2 { font-size: 17px; color: #4f46e5; margin: 0 0 12px; }
  .info-table { width: 100%; border-collapse: collapse; margin: 16px 0; }
  .info-table td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }
  .info-table td:first-child { font-weight: 600; color: #475569; width: 140px; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge-urgent { background: #fef2f2; color: #dc2626; }
  .badge-info { background: #eff6ff; color: #2563eb; }
  .cta { display: inline-block; margin: 20px 0 8px; padding: 10px 24px; background: #4f46e5; color: #ffffff; text-decoration: none; border-radius: 6px; font-size: 14px; font-weight: 600; }
  .footer { padding: 16px 32px; background: #f8fafc; border-top: 1px solid #e2e8f0; text-align: center; font-size: 12px; color: #94a3b8; }
  .conflict-box { border-left: 4px solid #f59e0b; background: #fffbeb; padding: 14px 16px; border-radius: 4px; margin: 14px 0; font-size: 14px; }
  .decision-old { border-left: 4px solid #ef4444; background: #fef2f2; padding: 12px 16px; border-radius: 4px; margin: 10px 0; font-size: 14px; }
  .decision-new { border-left: 4px solid #22c55e; background: #f0fdf4; padding: 12px 16px; border-radius: 4px; margin: 10px 0; font-size: 14px; }
</style>
"""


def _wrap_html(title: str, body_html: str) -> str:
    """Wrap body content in the full HTML email template."""
    return f"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {_BASE_STYLE}
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>{title}</h1>
      <p>Meeting Mind — Intelligent Meeting Assistant</p>
    </div>
    <div class="body">
      {body_html}
    </div>
    <div class="footer">
      This is an automated message from <strong>Meeting Mind</strong>. Please do not reply directly.
    </div>
  </div>
</body>
</html>"""


class EmailNotifier:
    """Sends transactional emails via SMTP (STARTTLS)."""

    def __init__(self) -> None:
        self._host = settings.smtp_host
        self._port = settings.smtp_port
        self._user = settings.smtp_user
        self._password = settings.smtp_password
        self._from = settings.smtp_from or settings.smtp_user

    # ── core send ─────────────────────────────────────────────────────────

    async def send_email(
        self,
        to: str,
        subject: str,
        html_body: str,
        text_body: str | None = None,
    ) -> bool:
        """Send a multipart email.  Returns ``True`` on success."""
        if not settings.smtp_configured:
            logger.warning("SMTP is not configured — email to %s skipped.", to)
            return False

        msg = MIMEMultipart("alternative")
        msg["From"] = self._from
        msg["To"] = to
        msg["Subject"] = subject

        if text_body:
            msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            smtp = aiosmtplib.SMTP(
                hostname=self._host,
                port=self._port,
                start_tls=True,
            )
            await smtp.connect()
            await smtp.login(self._user, self._password)
            await smtp.send_message(msg)
            await smtp.quit()
            logger.info("Email sent to %s — %s", to, subject)
            return True
        except Exception:
            logger.exception("Failed to send email to %s — %s", to, subject)
            return False

    # ── reminder ──────────────────────────────────────────────────────────

    async def send_reminder(
        self,
        to_email: str,
        action_item_desc: str,
        deadline: str,
        meeting_title: str,
        hours_remaining: int,
    ) -> bool:
        """Send a deadline-reminder email."""
        subject = (
            f"⏰ Reminder: Task due in {hours_remaining}h — "
            f"{action_item_desc[:50]}"
        )

        urgency_class = "badge-urgent" if hours_remaining <= 4 else "badge-info"
        urgency_label = "Urgent" if hours_remaining <= 4 else "Upcoming"

        body = f"""\
<h2>⏰ Task Deadline Reminder</h2>
<p>You have a task due soon. Here are the details:</p>
<table class="info-table">
  <tr><td>Task</td><td>{action_item_desc}</td></tr>
  <tr><td>Deadline</td><td><strong>{deadline}</strong></td></tr>
  <tr><td>Time Left</td><td><span class="badge {urgency_class}">{urgency_label} — {hours_remaining}h remaining</span></td></tr>
  <tr><td>Meeting</td><td>{meeting_title}</td></tr>
</table>
<p>Please make sure to complete this task before the deadline.</p>
<a class="cta" href="#">Open Dashboard</a>
"""
        html = _wrap_html("Task Reminder", body)
        text = (
            f"Reminder: \"{action_item_desc}\" is due in {hours_remaining} hours "
            f"(deadline: {deadline}).  From meeting: {meeting_title}."
        )
        return await self.send_email(to_email, subject, html, text)

    # ── conflict notification ─────────────────────────────────────────────

    async def send_conflict_notification(
        self,
        to_email: str,
        conflict_explanation: str,
        new_decision: str,
        old_decision: str,
        new_meeting: str,
        old_meeting: str,
    ) -> bool:
        """Notify a stakeholder about a detected decision conflict."""
        subject = "⚠️ Decision Conflict Detected — Action Required"

        body = f"""\
<h2>⚠️ Decision Conflict Detected</h2>
<p>A potential contradiction between two decisions has been identified and may need your attention.</p>

<div class="conflict-box">
  <strong>Explanation:</strong><br>{conflict_explanation}
</div>

<h2 style="margin-top:24px;">New Decision</h2>
<div class="decision-new">
  <strong>Meeting:</strong> {new_meeting}<br>
  <strong>Decision:</strong> {new_decision}
</div>

<h2 style="margin-top:24px;">Previous Decision</h2>
<div class="decision-old">
  <strong>Meeting:</strong> {old_meeting}<br>
  <strong>Decision:</strong> {old_decision}
</div>

<p style="margin-top:20px;">Please review the conflict on the dashboard and choose a resolution.</p>
<a class="cta" href="#">Review Conflict</a>
"""
        html = _wrap_html("Decision Conflict", body)
        text = (
            f"Decision conflict detected.\n\n"
            f"Explanation: {conflict_explanation}\n\n"
            f"New decision (from {new_meeting}): {new_decision}\n"
            f"Old decision (from {old_meeting}): {old_decision}\n\n"
            f"Please review and resolve."
        )
        return await self.send_email(to_email, subject, html, text)

    # ── weekly digest ─────────────────────────────────────────────────────

    async def send_digest(
        self,
        to_email: str,
        member_name: str,
        digest_html: str,
        week_start: str,
    ) -> bool:
        """Send a weekly meeting digest."""
        subject = f"📋 Your Weekly Meeting Digest — Week of {week_start}"

        body = f"""\
<h2>Hi {member_name} 👋</h2>
<p>Here's your meeting activity summary for the week starting <strong>{week_start}</strong>.</p>
<hr style="border: none; border-top: 1px solid #e2e8f0; margin: 20px 0;">
{digest_html}
<hr style="border: none; border-top: 1px solid #e2e8f0; margin: 20px 0;">
<p>Visit the dashboard for full details and to take action on pending items.</p>
<a class="cta" href="#">Open Dashboard</a>
"""
        html = _wrap_html("Weekly Digest", body)
        text = (
            f"Hi {member_name},\n\n"
            f"Here is your weekly meeting digest for the week of {week_start}.\n\n"
            f"Please check the dashboard for details."
        )
        return await self.send_email(to_email, subject, html, text)
