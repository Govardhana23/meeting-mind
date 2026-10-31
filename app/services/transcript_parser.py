"""
Transcript parser — converts raw meeting transcripts (plain text, VTT, SRT)
into speaker-attributed utterance dicts.
"""

from __future__ import annotations

import re
from typing import Optional


# ─── Timestamp helpers ───────────────────────────────────────────────────────

_TS_HH_MM_SS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?")
_TS_MM_SS = re.compile(r"(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?")


def _parse_timestamp(raw: str) -> Optional[float]:
    """Convert a HH:MM:SS.mmm or MM:SS.mmm string to seconds."""
    raw = raw.strip()
    m = _TS_HH_MM_SS.fullmatch(raw)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ms = int((m.group(4) or "0").ljust(3, "0"))
        return h * 3600 + mi * 60 + s + ms / 1000.0
    m = _TS_MM_SS.fullmatch(raw)
    if m:
        mi, s = int(m.group(1)), int(m.group(2))
        ms = int((m.group(3) or "0").ljust(3, "0"))
        return mi * 60 + s + ms / 1000.0
    return None


# ─── Speaker-name normalisation ─────────────────────────────────────────────

def normalize_speaker(name: str) -> str:
    """Strip extra whitespace and title-case a speaker name."""
    return " ".join(name.split()).strip().title()


# ─── Merge consecutive same-speaker utterances ──────────────────────────────

def merge_consecutive(utterances: list[dict]) -> list[dict]:
    """Merge consecutive utterances from the same speaker into one."""
    if not utterances:
        return []
    merged: list[dict] = [utterances[0].copy()]
    for utt in utterances[1:]:
        prev = merged[-1]
        if utt["speaker"] == prev["speaker"]:
            prev["text"] = prev["text"].rstrip() + " " + utt["text"].lstrip()
            prev["end_time"] = utt["end_time"]
        else:
            merged.append(utt.copy())
    # re-sequence
    for idx, item in enumerate(merged):
        item["sequence"] = idx
    return merged


# ─── Plain-text parser ──────────────────────────────────────────────────────

# Pattern: [HH:MM:SS] Speaker Name: text
_PLAIN_TS_BRACKET = re.compile(
    r"^\[(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\]\s+(.+?):\s+(.*)"
)
# Pattern: Speaker Name (HH:MM:SS): text
_PLAIN_SPEAKER_TS = re.compile(
    r"^(.+?)\s+\((\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\):\s+(.*)"
)
# Pattern: [Speaker Name]: text
_PLAIN_BRACKET_SPEAKER = re.compile(r"^\[(.+?)\]:\s+(.*)")
# Pattern: Speaker Name: text  (speaker can't start with a digit to avoid clashing with SRT numbers)
_PLAIN_SPEAKER = re.compile(r"^([A-Za-z][A-Za-z .'\-]{0,80}):\s+(.*)")


def _parse_plain(raw_text: str) -> list[dict]:
    """Parse plain-text transcripts with speaker labels.

    Supports:
      - ``[HH:MM:SS] Speaker Name: text``
      - ``Speaker Name (HH:MM:SS): text``
      - ``[Speaker Name]: text``
      - ``Speaker Name: text``

    If no speaker labels are detected the whole text is treated as a single
    utterance attributed to ``"Unknown"`` with ``needs_reattribution=True``.
    """
    lines = raw_text.strip().splitlines()
    utterances: list[dict] = []
    seq = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        speaker: Optional[str] = None
        text: Optional[str] = None
        start: Optional[float] = None

        # Try each pattern in order of specificity
        m = _PLAIN_TS_BRACKET.match(line)
        if m:
            start = _parse_timestamp(m.group(1))
            speaker = m.group(2)
            text = m.group(3)
        else:
            m = _PLAIN_SPEAKER_TS.match(line)
            if m:
                speaker = m.group(1)
                start = _parse_timestamp(m.group(2))
                text = m.group(3)
            else:
                m = _PLAIN_BRACKET_SPEAKER.match(line)
                if m:
                    speaker = m.group(1)
                    text = m.group(2)
                else:
                    m = _PLAIN_SPEAKER.match(line)
                    if m:
                        speaker = m.group(1)
                        text = m.group(2)

        if speaker is not None and text is not None:
            utterances.append({
                "speaker": normalize_speaker(speaker),
                "text": text.strip(),
                "start_time": start,
                "end_time": None,
                "sequence": seq,
            })
            seq += 1
        elif utterances:
            # continuation line – append to previous utterance
            utterances[-1]["text"] += " " + line
        # else: leading text before any speaker label → skip

    # If we found no speaker labels at all, treat entire text as one utterance
    if not utterances:
        utterances = [{
            "speaker": "Unknown",
            "text": raw_text.strip(),
            "start_time": None,
            "end_time": None,
            "sequence": 0,
            "needs_reattribution": True,
        }]

    return utterances


# ─── VTT parser ─────────────────────────────────────────────────────────────

_VTT_TS_LINE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s+-->\s+(\d{1,2}:\d{2}:\d{2}[.,]\d{3})"
)
_VTT_VOICE_TAG = re.compile(r"<v\s+([^>]+)>")


def _parse_vtt(raw_text: str) -> list[dict]:
    """Parse a WebVTT file with optional ``<v SpeakerName>`` voice tags."""
    lines = raw_text.strip().splitlines()
    utterances: list[dict] = []
    seq = 0
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_speaker: str = "Unknown"
    current_note_speaker: Optional[str] = None
    cue_lines: list[str] = []
    in_cue = False

    def _flush() -> None:
        nonlocal seq, cue_lines, current_start, current_end, current_speaker, current_note_speaker
        if cue_lines:
            full_text = " ".join(cue_lines)
            # extract speaker from <v ...> tag if present
            v_match = _VTT_VOICE_TAG.search(full_text)
            if v_match:
                speaker = v_match.group(1)
                # strip all voice tags from text
                clean = re.sub(r"<v\s+[^>]+>", "", full_text)
                clean = re.sub(r"</v>", "", clean).strip()
            elif current_note_speaker:
                speaker = current_note_speaker
                clean = full_text.strip()
            else:
                speaker = current_speaker
                clean = full_text.strip()

            # Strip any remaining HTML-like tags
            clean = re.sub(r"<[^>]+>", "", clean).strip()
            if clean:
                utterances.append({
                    "speaker": normalize_speaker(speaker),
                    "text": clean,
                    "start_time": current_start,
                    "end_time": current_end,
                    "sequence": seq,
                })
                seq += 1
            cue_lines = []
            current_note_speaker = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip the WEBVTT header and STYLE blocks
        if line.upper().startswith("WEBVTT") or line.upper().startswith("STYLE"):
            i += 1
            continue

        # NOTE blocks can carry speaker info  e.g. "NOTE speaker:John"
        if line.upper().startswith("NOTE"):
            note_text = line[4:].strip()
            # Check for "speaker:Name" in the NOTE
            if ":" in note_text:
                current_note_speaker = note_text.split(":", 1)[1].strip()
            else:
                # multi-line NOTE
                i += 1
                while i < len(lines) and lines[i].strip():
                    note_line = lines[i].strip()
                    if "speaker" in note_line.lower() and ":" in note_line:
                        current_note_speaker = note_line.split(":", 1)[1].strip()
                    i += 1
            i += 1
            continue

        # Timestamp line?
        ts_match = _VTT_TS_LINE.match(line)
        if ts_match:
            _flush()
            current_start = _parse_timestamp(ts_match.group(1))
            current_end = _parse_timestamp(ts_match.group(2))
            in_cue = True
            i += 1
            continue

        # Empty line ends a cue
        if not line:
            if in_cue:
                _flush()
                in_cue = False
            i += 1
            continue

        # Cue identifier lines (purely numeric or with a label before -->)
        if in_cue is False and line and not _VTT_TS_LINE.match(line):
            # Could be a cue id, skip it
            i += 1
            continue

        # Inside a cue, collect text lines
        if in_cue:
            cue_lines.append(line)

        i += 1

    _flush()
    return utterances


# ─── SRT parser ─────────────────────────────────────────────────────────────

_SRT_INDEX = re.compile(r"^\d+$")
_SRT_TS_LINE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,]\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}[,]\d{3})"
)
_SRT_SPEAKER_PREFIX = re.compile(r"^([A-Za-z][A-Za-z .'\-]{0,80}):\s+(.*)")


def _parse_srt(raw_text: str) -> list[dict]:
    """Parse an SRT file.  Speaker labels are detected as ``Speaker: text``
    prefixes in the subtitle text."""
    lines = raw_text.strip().splitlines()
    utterances: list[dict] = []
    seq = 0
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    cue_lines: list[str] = []
    state = "EXPECT_INDEX"

    def _flush() -> None:
        nonlocal seq, cue_lines, current_start, current_end
        if cue_lines:
            full_text = " ".join(cue_lines).strip()
            speaker = "Unknown"
            text = full_text

            # Check for Speaker: prefix in the combined text
            sp = _SRT_SPEAKER_PREFIX.match(full_text)
            if sp:
                speaker = sp.group(1)
                text = sp.group(2)

            if text:
                utterances.append({
                    "speaker": normalize_speaker(speaker),
                    "text": text,
                    "start_time": current_start,
                    "end_time": current_end,
                    "sequence": seq,
                })
                seq += 1
            cue_lines = []

    for line in lines:
        stripped = line.strip()

        if state == "EXPECT_INDEX":
            if _SRT_INDEX.match(stripped):
                state = "EXPECT_TS"
            elif not stripped:
                continue
            continue

        if state == "EXPECT_TS":
            ts_match = _SRT_TS_LINE.match(stripped)
            if ts_match:
                _flush()
                current_start = _parse_timestamp(ts_match.group(1))
                current_end = _parse_timestamp(ts_match.group(2))
                state = "EXPECT_TEXT"
            continue

        if state == "EXPECT_TEXT":
            if not stripped:
                _flush()
                state = "EXPECT_INDEX"
            else:
                cue_lines.append(stripped)
            continue

    _flush()
    return utterances


# ─── Public API ──────────────────────────────────────────────────────────────

def parse_transcript(raw_text: str, format: str = "plain") -> list[dict]:
    """Parse a raw transcript string into speaker-attributed utterances.

    Parameters
    ----------
    raw_text:
        The full transcript text.
    format:
        One of ``"plain"``, ``"vtt"``, ``"srt"``.

    Returns
    -------
    list[dict]
        Each dict contains:
        ``{"speaker": str, "text": str, "start_time": float|None,
           "end_time": float|None, "sequence": int}``

        If no speaker labels could be detected (plain-text only), the single
        returned utterance will also carry ``"needs_reattribution": True``.
    """
    fmt = format.lower().strip()
    if fmt == "vtt":
        utterances = _parse_vtt(raw_text)
    elif fmt == "srt":
        utterances = _parse_srt(raw_text)
    else:
        utterances = _parse_plain(raw_text)

    # Normalise speaker names already done per-parser; merge consecutive
    utterances = merge_consecutive(utterances)
    return utterances
