"""
Pydantic schemas for request / response validation.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional
from pydantic import BaseModel, Field, EmailStr


# ─── Team Members ────────────────────────────────────────────────────────────

class TeamMemberCreate(BaseModel):
    name: str
    email: EmailStr
    aliases: str = ""  # comma-separated alternative names


class TeamMemberOut(BaseModel):
    id: str
    name: str
    email: str
    aliases: str
    model_config = {"from_attributes": True}


# ─── Meeting Upload ─────────────────────────────────────────────────────────

class MeetingUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    transcript: str = Field(..., min_length=10)
    format: str = Field(default="plain", pattern="^(plain|vtt|srt)$")
    meeting_date: Optional[_dt.datetime] = None


class MeetingOut(BaseModel):
    id: str
    title: str
    meeting_date: _dt.datetime
    status: str
    created_at: _dt.datetime
    model_config = {"from_attributes": True}


class MeetingDetail(MeetingOut):
    raw_transcript: str
    utterances: list[UtteranceOut] = []
    decisions: list[DecisionOut] = []
    action_items: list[ActionItemOut] = []


# ─── Utterance ───────────────────────────────────────────────────────────────

class UtteranceOut(BaseModel):
    id: str
    speaker: str
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    sequence: int
    model_config = {"from_attributes": True}


# ─── Decision ───────────────────────────────────────────────────────────────

class DecisionOut(BaseModel):
    id: str
    text: str
    context: Optional[str] = None
    topic: Optional[str] = None
    decided_by: Optional[str] = None
    status: str
    created_at: _dt.datetime
    model_config = {"from_attributes": True}


# ─── Action Item ─────────────────────────────────────────────────────────────

class ActionItemOut(BaseModel):
    id: str
    description: str
    owner_name: str
    deadline: Optional[_dt.datetime] = None
    priority: str
    status: str
    tracker_card_id: Optional[str] = None
    tracker_card_url: Optional[str] = None
    created_at: _dt.datetime
    model_config = {"from_attributes": True}


class ActionItemUpdate(BaseModel):
    description: Optional[str] = None
    owner_name: Optional[str] = None
    deadline: Optional[_dt.datetime] = None
    priority: Optional[str] = None
    status: Optional[str] = None


# ─── Conflict ───────────────────────────────────────────────────────────────

class ConflictOut(BaseModel):
    id: str
    new_decision_id: str
    old_decision_id: str
    explanation: str
    severity: str
    resolution: Optional[str] = None
    status: str
    created_at: _dt.datetime
    # Enriched fields from joins
    new_decision_text: Optional[str] = None
    old_decision_text: Optional[str] = None
    new_meeting_title: Optional[str] = None
    old_meeting_title: Optional[str] = None
    model_config = {"from_attributes": True}


class ConflictResolve(BaseModel):
    resolution: str
    keep_decision: str = Field(default="new", pattern="^(new|old|both)$")


# ─── Pending Action ─────────────────────────────────────────────────────────

class PendingActionOut(BaseModel):
    id: str
    meeting_id: str
    action_type: str
    reference_id: str
    payload: Optional[str] = None
    status: str
    created_at: _dt.datetime
    model_config = {"from_attributes": True}


class PendingActionReview(BaseModel):
    status: str = Field(..., pattern="^(approved|rejected)$")
    edits: Optional[dict] = None  # optional field-level edits before approval


# ─── Digest ──────────────────────────────────────────────────────────────────

class DigestOut(BaseModel):
    member_name: str
    member_email: str
    week_start: _dt.date
    action_items: list[ActionItemOut]
    decisions_involved: list[DecisionOut]
    conflicts: list[ConflictOut]
    summary: str


# ─── Processing Result ──────────────────────────────────────────────────────

class ProcessingResult(BaseModel):
    meeting_id: str
    utterances_count: int
    decisions_count: int
    action_items_count: int
    conflicts_found: int
    pending_actions_count: int


# Forward-ref resolution
MeetingDetail.model_rebuild()
