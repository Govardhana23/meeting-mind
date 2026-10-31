"""
LLM-based extraction — uses Google GenAI (Gemini) to extract structured
decisions, action items, and re-attributed speaker turns from transcripts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from google import genai

from app.config import settings

logger = logging.getLogger(__name__)

_MODEL = "gemini-2.0-flash"


# ─── Client factory ─────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    """Return a configured GenAI client, raising if the API key is absent."""
    if not settings.gemini_configured:
        raise RuntimeError(
            "Gemini API key is not configured. "
            "Set the GEMINI_API_KEY environment variable or add it to .env."
        )
    return genai.Client(api_key=settings.gemini_api_key)


# ─── Retry wrapper ──────────────────────────────────────────────────────────

async def _call_gemini(client: genai.Client, prompt: str, *, max_retries: int = 3) -> str:
    """Call Gemini with exponential-backoff retries. Returns the text response."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(
                "Gemini API call failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1, max_retries, exc, wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError(
        f"Gemini API call failed after {max_retries} attempts"
    ) from last_exc


# ─── JSON extraction helper ─────────────────────────────────────────────────

def _extract_json(text: str) -> Any:
    """Extract a JSON object or array from a Gemini response that may be
    wrapped in markdown code fences."""
    # Try to find a JSON code block first
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    payload = m.group(1).strip() if m else text.strip()

    # Some models add trailing commas or comments; try parsing directly
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    # Fallback: locate the first { or [ and match to the last } or ]
    start_obj = payload.find("{")
    start_arr = payload.find("[")
    if start_obj == -1 and start_arr == -1:
        raise ValueError(f"No JSON found in LLM response:\n{text[:500]}")

    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
        end = payload.rfind("]")
        fragment = payload[start_arr: end + 1]
    else:
        end = payload.rfind("}")
        fragment = payload[start_obj: end + 1]

    return json.loads(fragment)


# ─── Build utterance transcript block ────────────────────────────────────────

def _format_utterances(utterances: list[dict]) -> str:
    """Format a list of utterance dicts into a readable transcript block."""
    lines: list[str] = []
    for u in utterances:
        ts = ""
        if u.get("start_time") is not None:
            minutes, secs = divmod(int(u["start_time"]), 60)
            hours, minutes = divmod(minutes, 60)
            ts = f"[{hours:02d}:{minutes:02d}:{secs:02d}] "
        lines.append(f"{ts}{u['speaker']}: {u['text']}")
    return "\n".join(lines)


# ─── Fallback Natural Language Extractor ────────────────────────────────────

def _rule_based_extraction(text_or_utterances: list[dict] | str, meeting_title: str, meeting_date: str) -> dict:
    """
    Intelligent NLP fallback extractor that extracts decisions and action items
    using grammatical cues, modal verbs, and speaker patterns.
    """
    from datetime import datetime, timedelta
    ref_date = datetime.utcnow().date()
    if meeting_date:
        try:
            ref_date = datetime.fromisoformat(meeting_date[:10]).date()
        except Exception:
            pass

    lines: list[tuple[str, str]] = []  # (speaker, text)
    if isinstance(text_or_utterances, list):
        for u in text_or_utterances:
            spk = u.get("speaker", "Unknown")
            full_text = u.get("text", "").strip()
            if full_text:
                lines.append((spk, full_text))
    else:
        for raw_line in text_or_utterances.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            m = re.match(r"^([A-Za-z][A-Za-z0-9 _\-]+):\s*(.*)", raw_line)
            if m:
                spk = m.group(1).strip()
                rest = m.group(2).strip()
            else:
                spk = "Unknown"
                rest = raw_line
            if rest:
                lines.append((spk, rest))

    decisions = []
    action_items = []

    decision_cues = [
        r"(?:we|team|everyone)\s+(?:decided|agreed|concluded)\s+(?:to|that)\s+(.+)",
        r"(?:let's|lets)\s+(?:go with|move forward with|adopt|choose|use)\s+(.+)",
        r"(?:the\s+)?final\s+decision\s+is\s+(?:to\s+)?(.+)",
        r"(?:approved|confirmed)\s+(?:the|to|that)\s+(.+)",
        r"(?:consensus\s+is\s+(?:to|that)\s+)(.+)",
    ]

    action_cues = [
        r"(?:will|gonna)\s+(?:take care of|prepare|finish|implement|deliver|complete|write|review|deploy|build|fix)\s+(.+)",
        r"(?:action item|todo|task):\s*(.+)",
        r"(?:please|kindly)\s+(?:make sure|handle|take the lead on|send)\s+(.+)",
        r"(?:assigned to|owner is)\s+([A-Za-z]+)\s*(?:to\s+)?(.+)?",
        r"(?:i'll|i will)\s+(.+)",
    ]

    for spk, text in lines:
        t_lower = text.lower()

        # Check for decisions
        for pattern in decision_cues:
            m = re.search(pattern, t_lower)
            if m:
                decisions.append({
                    "text": text.strip(),
                    "context": f"Discussed during {meeting_title}",
                    "topic": meeting_title,
                    "decided_by": spk if spk != "Unknown" else "Team",
                })
                break

        # Check for action items
        is_action = False
        action_desc = ""
        action_owner = spk if spk != "Unknown" else "Unassigned"
        
        for pattern in action_cues:
            m = re.search(pattern, t_lower)
            if m:
                is_action = True
                action_desc = text.strip()
                # Check for explicit person mention like "Bob, will you..."
                name_match = re.search(r"\b([A-Z][a-z]+)[,\s]+(?:can you|please|will handle)", text)
                if name_match:
                    action_owner = name_match.group(1)
                break

        if is_action:
            # Deadline extraction
            dl_raw = ""
            dl_iso = ""
            if "by friday" in t_lower:
                dl_raw = "by Friday"
                days_ahead = (4 - ref_date.weekday() + 7) % 7 or 7
                dl_iso = (ref_date + timedelta(days=days_ahead)).isoformat()
            elif "by monday" in t_lower:
                dl_raw = "by Monday"
                days_ahead = (0 - ref_date.weekday() + 7) % 7 or 7
                dl_iso = (ref_date + timedelta(days=days_ahead)).isoformat()
            elif "by tomorrow" in t_lower:
                dl_raw = "by tomorrow"
                dl_iso = (ref_date + timedelta(days=1)).isoformat()
            elif "next week" in t_lower:
                dl_raw = "next week"
                dl_iso = (ref_date + timedelta(days=7)).isoformat()
            elif "end of sprint" in t_lower:
                dl_raw = "end of sprint"
                dl_iso = (ref_date + timedelta(days=14)).isoformat()

            # Priority extraction
            priority = "medium"
            if any(w in t_lower for w in ["urgent", "asap", "blocker", "critical", "immediately"]):
                priority = "critical"
            elif any(w in t_lower for w in ["high priority", "important", "must have", "tight deadline"]):
                priority = "high"
            elif any(w in t_lower for w in ["low priority", "nice to have", "when you have time", "optional"]):
                priority = "low"

            action_items.append({
                "description": action_desc,
                "owner_name": action_owner,
                "owner": action_owner,
                "deadline_raw": dl_raw,
                "deadline": dl_iso,
                "deadline_iso": dl_iso,
                "priority": priority,
                "topic": meeting_title,
            })

    # If no decisions found, look for general agreement markers
    if not decisions and lines:
        for spk, text in lines:
            if any(w in text.lower() for w in ["agreed", "decided", "let's use", "going with", "approved", "we will stick to"]):
                decisions.append({
                    "text": text.strip(),
                    "context": f"Agreed in {meeting_title}",
                    "topic": meeting_title,
                    "decided_by": spk,
                })

    return {"decisions": decisions, "action_items": action_items}


# ─── Extract decisions & action items ────────────────────────────────────────

async def extract_decisions_and_actions(
    utterances: list[dict] | str,
    meeting_title: str,
    meeting_date: str = "",
) -> dict:
    """Use Gemini (or intelligent fallback) to extract decisions and action items."""
    # Normalize utterances to transcript block
    if isinstance(utterances, str):
        transcript_block = utterances
    else:
        transcript_block = _format_utterances(utterances)

    if not meeting_date:
        meeting_date = datetime.utcnow().date().isoformat()

    if not settings.gemini_configured:
        return _rule_based_extraction(utterances, meeting_title, meeting_date)

    prompt = f"""You are an expert meeting-notes analyst.

**Meeting title:** {meeting_title}
**Meeting date:** {meeting_date}

Below is the meeting transcript.  Analyse it carefully and return a JSON object with two keys: `decisions` and `action_items`.

### TRANSCRIPT
{transcript_block}

### INSTRUCTIONS

**decisions** — an array of objects, each with:
- `text` (string): the precise decision that was made.
- `context` (string): a 1–2 sentence summary of the surrounding discussion that led to this decision.
- `topic` (string): a short topic label (e.g. "API design", "Hiring").
- `decided_by` (string): the person(s) who explicitly made or approved the decision.

Rules for decisions:
• Only include statements that are clearly *agreed* or *decided*, not things merely discussed, proposed, or debated without resolution.
• If someone says "Let's go with X" and no one objects, that counts as a decision.
• Capture the decision-maker's name exactly as it appears in the transcript.

**action_items** — an array of objects, each with:
- `description` (string): what needs to be done.
- `owner_name` (string): the person responsible. Use the name as it appears in the transcript. If unclear, write "Unassigned".
- `deadline_raw` (string): the original deadline phrase from the transcript (e.g. "by Friday", "next week", "end of sprint"). If none mentioned, use "".
- `deadline_iso` (string): the deadline resolved to an ISO-8601 date (YYYY-MM-DD) relative to the meeting date {meeting_date}. If you cannot determine a date, use "".
  Resolution guidance:
    - "by Friday" → the next Friday on or after {meeting_date}.
    - "next week" → the Monday of the following week.
    - "end of sprint" → 2 weeks from {meeting_date}.
    - "ASAP" / "immediately" → {meeting_date}.
    - "end of month" → last day of the month of {meeting_date}.
    - "end of quarter" → last day of the current quarter.
- `priority` (string): one of "low", "medium", "high", "critical".
  Priority guidance:
    - "critical" if words like "urgent", "blocker", "critical", "ASAP" are used.
    - "high" if the task is explicitly important or has a tight deadline.
    - "medium" for standard follow-up tasks.
    - "low" for nice-to-haves, future ideas, or non-blocking tasks.

Rules for action items:
• Include both explicit ("Alice, please do X") and implicit ("We should do Y" — attribute to the speaker or "Unassigned").
• Do NOT include decisions that are merely informational with no follow-up work.

Return ONLY the JSON object — no commentary, no markdown fences.

Example output:
{{"decisions": [{{"text": "...", "context": "...", "topic": "...", "decided_by": "..."}}], "action_items": [{{"description": "...", "owner_name": "...", "deadline_raw": "...", "deadline_iso": "...", "priority": "..."}}]}}
"""

    try:
        client = _get_client()
        raw = await _call_gemini(client, prompt)
        data = _extract_json(raw)

        # Normalise to expected shape
        if isinstance(data, dict):
            decisions = data.get("decisions", [])
            action_items = data.get("action_items", [])
        else:
            decisions = []
            action_items = []

        # Ensure every item has the required keys
        for d in decisions:
            d.setdefault("text", "")
            d.setdefault("context", "")
            d.setdefault("topic", "")
            d.setdefault("decided_by", "")

        for a in action_items:
            d_owner = a.get("owner_name") or a.get("owner") or "Unassigned"
            a["owner_name"] = d_owner
            a.setdefault("owner", d_owner)
            a.setdefault("description", "")
            a.setdefault("deadline_raw", "")
            dl_iso = a.get("deadline_iso") or a.get("deadline") or ""
            a["deadline_iso"] = dl_iso
            a.setdefault("deadline", dl_iso)
            a.setdefault("priority", "medium")
            # Clamp priority to allowed values
            if a["priority"] not in ("low", "medium", "high", "critical"):
                a["priority"] = "medium"

        return {"decisions": decisions, "action_items": action_items}
    except Exception as exc:
        logger.warning("Gemini extraction encountered an error: %s — using rule-based extraction fallback", exc)
        return _rule_based_extraction(utterances, meeting_title, meeting_date)


# ─── Speaker re-attribution ─────────────────────────────────────────────────

async def reattribute_speakers(
    text: str,
    known_names: list[str],
) -> list[dict]:
    """Use Gemini to infer speaker identities for an unlabelled transcript.

    Parameters
    ----------
    text:
        The raw transcript text without speaker labels.
    known_names:
        A list of known participant names to guide attribution.

    Returns
    -------
    list[dict]
        Utterances in the standard format:
        ``[{{"speaker": ..., "text": ..., "start_time": None,
              "end_time": None, "sequence": int}}]``
    """
    client = _get_client()

    names_block = ", ".join(known_names) if known_names else "(no known names — infer from context)"

    prompt = f"""You are an expert transcript editor.

The following meeting transcript has NO speaker labels.  Your task is to split it into speaker-attributed turns.

**Known participants:** {names_block}

### TRANSCRIPT
{text}

### INSTRUCTIONS
1. Identify turn boundaries where a different person starts speaking.
2. Attribute each turn to a speaker.  Use the known participant names when possible; if you must invent a name, use "Speaker 1", "Speaker 2", etc.
3. Return a JSON array of objects, each with:
   - `speaker` (string)
   - `text` (string): the verbatim text of that turn.
4. Preserve the original order.
5. Return ONLY the JSON array — no commentary.

Example:
[{{"speaker": "Alice", "text": "Good morning, let's start."}}, {{"speaker": "Bob", "text": "Sure, I have the report ready."}}]
"""

    raw = await _call_gemini(client, prompt)
    items = _extract_json(raw)

    if not isinstance(items, list):
        # Fallback: return original as single Unknown speaker
        return [{
            "speaker": "Unknown",
            "text": text.strip(),
            "start_time": None,
            "end_time": None,
            "sequence": 0,
        }]

    utterances: list[dict] = []
    for idx, item in enumerate(items):
        utterances.append({
            "speaker": str(item.get("speaker", "Unknown")).strip().title(),
            "text": str(item.get("text", "")).strip(),
            "start_time": None,
            "end_time": None,
            "sequence": idx,
        })

    return utterances
