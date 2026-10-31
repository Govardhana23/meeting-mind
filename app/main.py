"""
Meeting Mind — FastAPI application entry point.

Wires together:
- Database initialization
- API routers (meetings, confirmations, dashboard, pages)
- Background scheduler for reminders & digests
- Static/template serving
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db, async_session_factory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("meeting_mind")


# ─── Lifespan (startup / shutdown) ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info("Initializing database …")
    await init_db()

    # ── Start background scheduler ──
    scheduler = None
    try:
        from app.services.reminder_scheduler import setup_scheduler
        scheduler = setup_scheduler(async_session_factory)
        if scheduler and not scheduler.running:
            scheduler.start()
        logger.info("Background reminder scheduler active.")
    except Exception:
        logger.warning("Reminder scheduler not started (check configuration).", exc_info=True)

    # ── Config diagnostics ──
    logger.info("Gemini API configured: %s", settings.gemini_configured)
    logger.info("Trello configured:     %s", settings.trello_configured)
    logger.info("SMTP configured:       %s", settings.smtp_configured)

    yield  # ← app is running

    # ── Shutdown ──
    if scheduler:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down.")


# ─── App factory ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Meeting Mind",
        description="Cross-meeting intelligence: decisions, conflicts, actions, and memory.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — wide-open for local dev; tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register routers ─────────────────────────────────────────────────
    from app.routers.meetings import router as meetings_router
    from app.routers.confirmations import router as confirmations_router
    from app.routers.dashboard import router as dashboard_router
    from app.routers.pages import router as pages_router

    app.include_router(meetings_router)
    app.include_router(confirmations_router)
    app.include_router(dashboard_router)
    app.include_router(pages_router)

    return app


app = create_app()
