"""
Cross-meeting conflict detection service.

Detects contradictions, reversals, and shifts between newly made decisions
and past decisions stored in memory across different meetings.

Combines:
1. Dense vector embeddings (Gemini text-embedding-004 with fallback normalized TF-IDF semantic vector)
2. Cosine similarity candidate retrieval (semantic proximity filter)
3. LLM verification (Gemini prompt assessing mutual exclusivity, timeline clash, or strategy reversal)
4. Deterministic contradiction rules (topic clash, polarity inversion, alternative entity replacement)
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Sequence, Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import Decision, Meeting

logger = logging.getLogger("meeting_mind.conflicts")


# ─── Embedding Generation ───────────────────────────────────────────────────

def _compute_fallback_embedding(text: str, dim: int = 256) -> list[float]:
    """
    Deterministic pseudo-semantic embedding for fallback / offline execution.
    Normalizes character trigrams and token stems into a unit sphere vector.
    """
    vec = [0.0] * dim
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = clean.split()
    
    # Word hashing & character trigram hashing
    for word in words:
        h = abs(hash(word)) % dim
        vec[h] += 1.5
        for i in range(len(word) - 2):
            tri = word[i:i+3]
            th = abs(hash(tri)) % dim
            vec[th] += 0.5

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


async def compute_embedding(text: str) -> list[float]:
    """Compute dense embedding using Gemini or deterministic semantic fallback."""
    if not text:
        return [0.0] * 256

    if settings.gemini_configured:
        try:
            from google import genai
            client = genai.Client(api_key=settings.gemini_api_key)
            result = await client.aio.models.embed_content(
                model="text-embedding-004",
                contents=text,
            )
            if hasattr(result, "embeddings") and result.embeddings:
                return list(result.embeddings[0].values)
        except Exception as e:
            logger.warning("Gemini embedding call failed, using semantic fallback: %s", e)

    return _compute_fallback_embedding(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors using pure math."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ─── Deterministic Conflict Detection Rules ─────────────────────────────────

_NEGATION_TERMS = {
    "not", "no", "never", "cancel", "abandon", "drop", "stop", "halt",
    "discontinue", "scrap", "reject", "replace", "switch", "instead", "revert"
}

_OPPOSING_PAIRS = [
    ("remote", "in-office"),
    ("remote", "on-site"),
    ("wfh", "office"),
    ("cloud", "on-premise"),
    ("aws", "gcp"),
    ("postgres", "mongodb"),
    ("postgresql", "mongodb"),
    ("react", "vue"),
    ("weekly", "daily"),
    ("monthly", "bi-weekly"),
    ("free", "paid"),
    ("open source", "proprietary"),
    ("buy", "build"),
    ("hire", "freeze"),
    ("increase", "decrease"),
    ("delay", "accelerate"),
    ("postpone", "expedite"),
]


def _rule_based_conflict_check(new_text: str, old_text: str, new_topic: str = "", old_topic: str = "") -> tuple[bool, str, str]:
    """
    Heuristic contradiction & reversal detection when LLM is unavailable or for pre-filtering.
    Returns (is_conflict, explanation, severity).
    """
    n_lower = new_text.lower()
    o_lower = old_text.lower()
    
    # 1. Direct Opposing Pairs check
    for a, b in _OPPOSING_PAIRS:
        if (a in n_lower and b in o_lower) or (b in n_lower and a in o_lower):
            # Check if they share surrounding context or topic words
            n_words = set(re.findall(r"\w{4,}", n_lower))
            o_words = set(re.findall(r"\w{4,}", o_lower))
            overlap = n_words.intersection(o_words)
            if overlap or (new_topic and old_topic and new_topic.lower() == old_topic.lower()):
                return True, f"Opposing directives detected ('{a}' vs '{b}') regarding the same topic.", "high"

    # 2. Reversal or Replacement phrases in new decision
    replacement_match = re.search(r"(?:instead of|replacing|switch(?:ed)? from|switched to|migrat(?:e|ing) from)\s+([a-zA-Z0-9_\- ]+)", n_lower)
    if replacement_match:
        target = replacement_match.group(1).strip()
        if target and (target in o_lower or (old_topic and target in old_topic.lower())):
            return True, f"New decision explicitly replaces earlier choice '{target}'.", "high"

    # 3. Cancellation / Abandonment of earlier commitment
    for neg in ["cancel", "abandon", "drop", "postpone", "halt", "scrapped", "pause"]:
        if f"{neg} " in n_lower or f" {neg}" in n_lower:
            o_nouns = set(re.findall(r"\b[a-zA-Z]{4,}\b", o_lower)) - {"that", "with", "from", "this", "have", "will", "been"}
            n_nouns = set(re.findall(r"\b[a-zA-Z]{4,}\b", n_lower)) - {"that", "with", "from", "this", "have", "will", "been"}
            shared = o_nouns.intersection(n_nouns)
            if len(shared) >= 2:
                return True, f"New decision appears to cancel or halt an earlier decision concerning: {', '.join(list(shared)[:3])}.", "medium"

    return False, "", "low"


# ─── LLM Verification ───────────────────────────────────────────────────────

async def verify_conflicts_with_llm(
    new_decision_text: str,
    candidates: list[dict],
) -> list[dict]:
    """
    Ask Gemini to verify whether the new decision genuinely contradicts or reverses
    any of the candidate historical decisions.
    """
    if not candidates:
        return []

    if not settings.gemini_configured:
        # Fall back to heuristic rule verification
        verified = []
        for cand in candidates:
            is_conf, expl, sev = _rule_based_conflict_check(
                new_decision_text,
                cand.get("decision_text", ""),
                cand.get("topic", ""),
                cand.get("cand_topic", ""),
            )
            if is_conf or cand.get("similarity", 0) > 0.85:
                verified.append({
                    "old_decision_id": cand["decision_id"],
                    "explanation": expl or f"Substantial divergence on topic: {cand.get('decision_text', '')[:100]}",
                    "severity": sev if is_conf else "medium",
                })
        return verified

    from google import genai
    client = genai.Client(api_key=settings.gemini_api_key)

    candidates_formatted = "\n".join([
        f"ID: {c['decision_id']}\nMeeting: {c.get('meeting_title', 'Previous Meeting')}\nDecision: {c['decision_text']}\n"
        for c in candidates
    ])

    prompt = f"""You are a cross-meeting governance and memory intelligence system.
Analyze whether the NEW DECISION below contradicts, reverses, supersedes, or clashes with any of the PAST DECISIONS.

NEW DECISION:
"{new_decision_text}"

PAST DECISIONS CANDIDATES:
{candidates_formatted}

Analyze each candidate. A CONFLICT means:
- The new decision reverses, cancels, or directly contradicts the past decision.
- The new decision introduces a mutually exclusive policy or technology (e.g. Postgres vs Mongo, remote vs office).
- The new decision changes a finalized architecture or budget that conflicts with the prior commitment.
Complementary or additive decisions that simply build upon the past are NOT conflicts.

Return a JSON array of verified conflicts ONLY. If none conflict, return an empty array [].
Schema:
[
  {{
    "old_decision_id": "<ID from above>",
    "explanation": "<Clear reason why these two decisions clash>",
    "severity": "<low|medium|high>"
  }}
]
Return ONLY JSON.
"""
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        text = response.text.strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        payload = m.group(1).strip() if m else text
        res = json.loads(payload)
        if isinstance(res, list):
            return res
    except Exception as e:
        logger.warning("LLM conflict verification failed, falling back to rules: %s", e)

    # Heuristic fallback if LLM call had an issue
    verified = []
    for cand in candidates:
        is_conf, expl, sev = _rule_based_conflict_check(
            new_decision_text,
            cand.get("decision_text", ""),
            cand.get("topic", ""),
        )
        if is_conf:
            verified.append({
                "old_decision_id": cand["decision_id"],
                "explanation": expl,
                "severity": sev,
            })
    return verified


# ─── Main Pipeline: detect_conflicts ────────────────────────────────────────

async def detect_conflicts(
    session: AsyncSession,
    new_decisions: Sequence[Decision] | list[dict],
    meeting_id: Optional[str] = None,
) -> list[dict]:
    """
    Compare new decisions against all past active decisions stored in memory.
    
    Returns a list of verified conflict dictionaries:
    [
        {
            "new_decision_id": str,
            "old_decision_id": str,
            "explanation": str,
            "severity": "low" | "medium" | "high"
        }
    ]
    """
    # 1. Fetch all existing active decisions from other meetings
    query = (
        select(Decision)
        .options(selectinload(Decision.meeting))
        .where(Decision.status == "active")
    )
    if meeting_id:
        query = query.where(Decision.meeting_id != meeting_id)
    elif new_decisions and isinstance(new_decisions[0], Decision) and new_decisions[0].meeting_id:
        query = query.where(Decision.meeting_id != new_decisions[0].meeting_id)

    res = await session.execute(query)
    past_decisions = list(res.scalars().all())

    if not past_decisions or not new_decisions:
        return []

    # Ensure past decisions have embeddings in memory
    for pd in past_decisions:
        if not pd.embedding_json and pd.text:
            emb = await compute_embedding(pd.text)
            pd.embedding_json = json.dumps(emb)

    all_conflicts: list[dict] = []

    for nd in new_decisions:
        nd_id = nd.id if isinstance(nd, Decision) else nd.get("id")
        nd_text = nd.text if isinstance(nd, Decision) else nd.get("text", "")
        nd_topic = (nd.topic if isinstance(nd, Decision) else nd.get("topic")) or ""

        if not nd_text:
            continue

        # Get embedding for new decision
        if isinstance(nd, Decision) and nd.embedding_json:
            try:
                nd_emb = json.loads(nd.embedding_json)
            except Exception:
                nd_emb = await compute_embedding(nd_text)
        else:
            nd_emb = await compute_embedding(nd_text)

        # 2. Filter candidates based on vector similarity or topic overlap
        candidates: list[dict] = []
        for pd in past_decisions:
            sim = 0.0
            if pd.embedding_json:
                try:
                    pd_emb = json.loads(pd.embedding_json)
                    sim = cosine_similarity(nd_emb, pd_emb)
                except Exception:
                    sim = 0.0

            # Topic keyword match boost
            topic_match = False
            if nd_topic and pd.topic and nd_topic.lower() == pd.topic.lower():
                topic_match = True

            # If semantic similarity is > 0.55 or strong topic overlap, consider as candidate
            if sim >= 0.55 or topic_match or _rule_based_conflict_check(nd_text, pd.text, nd_topic, pd.topic or "")[0]:
                candidates.append({
                    "decision_id": pd.id,
                    "decision_text": pd.text,
                    "cand_topic": pd.topic,
                    "similarity": sim,
                    "meeting_title": pd.meeting.title if pd.meeting else "Previous Meeting",
                })

        if not candidates:
            continue

        # 3. Verify candidates with LLM reasoning / contradiction rules
        verified = await verify_conflicts_with_llm(nd_text, candidates)

        for v in verified:
            all_conflicts.append({
                "new_decision_id": nd_id,
                "old_decision_id": v["old_decision_id"],
                "explanation": v.get("explanation", "Contradicts previous decision made in earlier meeting."),
                "severity": v.get("severity", "medium"),
            })

    return all_conflicts
