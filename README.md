# Meeting Mind 🧠
### Cross-Meeting Intelligence & Governance System

Meeting Mind is a production-grade cross-meeting intelligence platform that connects organizational memory across meetings, detects contradictions between new and historical decisions, manages task tracker dispatch with human confirmation, and automates deadline reminders and weekly digests.

---

## Key Features

1. **Transcript Ingestion with Speaker Attribution**
   - Supports **Plain Text** (with timestamps and speaker prefixes), **WebVTT** (`.vtt`), and **SubRip** (`.srt`).
   - Normalizes speaker names, resolves timestamps, and merges consecutive turns.
   - LLM-assisted re-attribution for unlabelled transcripts.

2. **Extraction of Decisions, Action Items, Owners & Deadlines**
   - Distinguishes hard commitments and consensus agreements from conversational chatter.
   - Parses relative deadlines (*"by Friday"*, *"next week"*, *"end of sprint"*) into ISO-8601 calendar dates.
   - Assigns task priority (*critical, high, medium, low*) based on urgency cues.

3. **Cross-Meeting Memory & Conflict Detection**
   - Persistent decision memory store in SQLite with dense vector embeddings.
   - Queries historical decisions across previous meetings using cosine similarity and topic overlap.
   - Detects contradictions, strategy reversals, and mutually exclusive directives (e.g. *PostgreSQL vs MongoDB*, *in-office vs remote*).
   - Allows formal reconciliation in corporate memory (supersede past, adopt new, or reconcile).

4. **Tracker Integration (Trello & In-App)**
   - REST integration with Trello API to create cards with rich markdown descriptions, due dates, and member assignments.
   - Local tracker link fallback when external API keys are not supplied.

5. **Confirmation Step Before Posting**
   - All extracted action items, conflict notifications, and reminders are staged in a **Confirmation Queue** (`PendingAction`).
   - Users can review, edit (change description, reassign owner, modify deadline, adjust priority), and approve or reject items before anything is posted or sent.
   - Supports single-item and bulk-meeting approval.

6. **Deadline Reminders**
   - Background scheduler checks upcoming deadlines against configurable intervals (e.g., 24h, 2h before).
   - Stages reminder notifications in the confirmation queue to prevent spam.
   - Dispatches formatted HTML reminder emails via `aiosmtplib` with STARTTLS.

7. **Weekly Digest Per Person**
   - Aggregates each team member's active tasks, involved decisions, and unresolved conflicts from the past 7 days.
   - Generates an executive natural-language HTML summary.
   - Accessible via the UI preview and automated batch email dispatch.

---

## Project Structure

```
meeting-mind/
├── app/
│   ├── config.py                 # Typed configuration & environment loading
│   ├── database.py               # SQLAlchemy async models & SQLite engine
│   ├── models.py                 # Pydantic schemas for API validation
│   ├── main.py                   # FastAPI app factory & lifespan
│   ├── memory/
│   │   ├── __init__.py
│   │   └── decision_store.py     # Cross-meeting vector memory & history
│   ├── services/
│   │   ├── transcript_parser.py  # Plain-text, WebVTT, and SRT parsing
│   │   ├── llm_extractor.py      # Decision/action extractor (Gemini + fallback)
│   │   ├── conflict_detector.py  # Embedding cosine similarity & conflict logic
│   │   ├── task_tracker.py       # Trello API integration
│   │   ├── notifier.py           # aiosmtplib HTML email notifications
│   │   ├── reminder_scheduler.py # APScheduler deadline reminder engine
│   │   └── digest_generator.py   # Weekly per-person digest compiler
│   ├── routers/
│   │   ├── meetings.py           # Ingestion & background processing API
│   │   ├── confirmations.py      # Confirmation queue & human-in-the-loop review
│   │   ├── dashboard.py          # Stats, conflicts resolution, team API
│   │   └── pages.py              # Jinja2 HTML web interface routes
│   └── templates/
│       ├── base.html             # Tailwind CSS & navigation sidebar
│       ├── dashboard.html        # System overview & stats cards
│       ├── upload.html           # File dropzone & live polling progress
│       ├── meeting_detail.html   # Speaker turns, decisions, tasks, conflicts
│       ├── confirmations.html    # Editable confirmation review queue
│       ├── conflicts.html        # Memory contradiction reconciliation
│       ├── team.html             # Member management & alias mapping
│       └── digests.html          # Individual weekly digest previews
├── tests/
│   ├── test_meeting_mind.py      # Full 7-feature end-to-end test suite
│   └── test_routers.py           # FastAPI HTTP endpoints & UI verification
├── requirements.txt
├── run.py                        # Uvicorn launcher
└── .env.example
```

---

## Running the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your `GEMINI_API_KEY`, `TRELLO_*`, and `SMTP_*` if available.
*(Note: The system contains built-in semantic embedding fallbacks and local tracker links, so it works completely out of the box even without external API credentials.)*

### 3. Launch the Server
```bash
python run.py
```
Open your browser to:
- **Dashboard:** [http://localhost:8000/](http://localhost:8000/)
- **Ingest Transcript:** [http://localhost:8000/upload](http://localhost:8000/upload)
- **Confirmation Queue:** [http://localhost:8000/confirmations](http://localhost:8000/confirmations)
- **Cross-Meeting Conflicts:** [http://localhost:8000/conflicts](http://localhost:8000/conflicts)
- **Weekly Digests:** [http://localhost:8000/digests](http://localhost:8000/digests)

### 4. Run the Verification Test Suites
```bash
python tests/test_meeting_mind.py
python tests/test_routers.py
```
