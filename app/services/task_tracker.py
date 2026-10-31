"""
Trello integration for creating and managing task cards.

Provides a ``TrelloClient`` that wraps the Trello REST API via httpx,
plus a convenience function to create cards from ActionItem rows.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import ActionItem, Meeting, Decision

logger = logging.getLogger(__name__)

_TRELLO_BASE = "https://api.trello.com/1"


class TrelloClient:
    """Async wrapper around the Trello REST API."""

    def __init__(self) -> None:
        self._key = settings.trello_api_key
        self._token = settings.trello_token
        self._board_id = settings.trello_board_id
        self._list_id = settings.trello_list_id

    # ── helpers ───────────────────────────────────────────────────────────

    def _auth_params(self) -> dict[str, str]:
        return {"key": self._key, "token": self._token}

    def _is_configured(self) -> bool:
        return settings.trello_configured

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> dict | list:
        """Execute an authenticated Trello API request."""
        url = f"{_TRELLO_BASE}{path}"
        merged_params = {**self._auth_params(), **(params or {})}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method,
                url,
                params=merged_params,
                json=json_body,
            )
            response.raise_for_status()
            return response.json()

    # ── public API ────────────────────────────────────────────────────────

    async def create_card(
        self,
        name: str,
        description: str,
        due_date: str | None,
        member_email: str | None,
        labels: list[str] | None = None,
    ) -> dict:
        """Create a card on the configured Trello list.

        Returns ``{card_id, card_url, card_name}`` or an empty dict when
        Trello is not configured.
        """
        if not self._is_configured():
            logger.warning("Trello is not configured — skipping card creation.")
            return {}

        payload: dict = {
            "name": name,
            "desc": description,
            "idList": self._list_id,
        }
        if due_date:
            payload["due"] = due_date
        if labels:
            payload["idLabels"] = ",".join(labels)

        # Resolve member
        if member_email:
            member_id = await self.find_member_by_email(member_email)
            if member_id:
                payload["idMembers"] = member_id

        data = await self._request("POST", "/cards", params=payload)

        return {
            "card_id": data["id"],
            "card_url": data["shortUrl"],
            "card_name": data["name"],
        }

    async def update_card(self, card_id: str, updates: dict) -> dict:
        """Update an existing card.

        Supported *updates* keys: ``name``, ``desc``, ``due``, ``closed``,
        ``idMembers``.  Returns the updated card JSON or empty dict.
        """
        if not self._is_configured():
            logger.warning("Trello is not configured — skipping card update.")
            return {}

        allowed_keys = {"name", "desc", "due", "closed", "idMembers"}
        filtered = {k: v for k, v in updates.items() if k in allowed_keys}
        if not filtered:
            return {}

        data = await self._request("PUT", f"/cards/{card_id}", params=filtered)
        return data

    async def get_card(self, card_id: str) -> dict:
        """Fetch full card details.  Returns empty dict when not configured."""
        if not self._is_configured():
            logger.warning("Trello is not configured — skipping card fetch.")
            return {}

        data = await self._request("GET", f"/cards/{card_id}")
        return data

    async def add_comment(self, card_id: str, text: str) -> None:
        """Post a comment on a card."""
        if not self._is_configured():
            logger.warning("Trello is not configured — skipping comment.")
            return

        await self._request(
            "POST",
            f"/cards/{card_id}/actions/comments",
            params={"text": text},
        )

    async def get_board_members(self) -> list[dict]:
        """Return ``[{id, full_name, username, email}]`` for board members.

        Trello only exposes ``email`` for members whose profile is public,
        so the field may be ``None``.
        """
        if not self._is_configured():
            logger.warning("Trello is not configured — skipping member list.")
            return []

        raw: list[dict] = await self._request(  # type: ignore[assignment]
            "GET", f"/boards/{self._board_id}/members"
        )

        members: list[dict] = []
        for m in raw:
            members.append(
                {
                    "id": m.get("id"),
                    "full_name": m.get("fullName", ""),
                    "username": m.get("username", ""),
                    "email": m.get("email"),  # may be None
                }
            )
        return members

    async def find_member_by_email(self, email: str) -> str | None:
        """Look up a Trello member ID by email on the configured board.

        Returns the member ``id`` or ``None`` if not found.
        """
        members = await self.get_board_members()
        email_lower = email.lower()
        for m in members:
            if m.get("email") and m["email"].lower() == email_lower:
                return m["id"]
        return None


# ─── Standalone convenience ──────────────────────────────────────────────────


async def create_task_for_action_item(
    action_item: ActionItem,
    session: AsyncSession,
) -> dict | None:
    """Create a Trello card for an ActionItem and persist the card link.

    Returns the card info dict or ``None`` when Trello is not configured.
    """
    if not settings.trello_configured:
        logger.info("Trello not configured — no card created for action item %s.", action_item.id)
        return None

    # Build a rich description
    meeting: Meeting | None = None
    decision: Decision | None = None

    if action_item.meeting_id:
        result = await session.execute(select(Meeting).where(Meeting.id == action_item.meeting_id))
        meeting = result.scalar_one_or_none()

    if action_item.decision_id:
        result = await session.execute(select(Decision).where(Decision.id == action_item.decision_id))
        decision = result.scalar_one_or_none()

    lines = [
        f"**Task:** {action_item.description}",
        f"**Owner:** {action_item.owner_name}",
        f"**Priority:** {action_item.priority}",
    ]
    if action_item.deadline:
        lines.append(f"**Deadline:** {action_item.deadline.strftime('%Y-%m-%d %H:%M UTC')}")
    if meeting:
        lines.append(f"**Meeting:** {meeting.title} ({meeting.meeting_date.strftime('%Y-%m-%d')})")
    if decision:
        lines.append(f"**Decision:** {decision.text}")

    description = "\n".join(lines)

    due_date: str | None = None
    if action_item.deadline:
        due_date = action_item.deadline.isoformat()

    # Resolve email for member assignment
    member_email: str | None = None
    if action_item.owner_member:
        member_email = action_item.owner_member.email

    client = TrelloClient()
    card = await client.create_card(
        name=action_item.description[:500],
        description=description,
        due_date=due_date,
        member_email=member_email,
    )

    if card:
        action_item.tracker_card_id = card["card_id"]
        action_item.tracker_card_url = card["card_url"]
        session.add(action_item)
        await session.commit()

    return card or None
