"""
Test FastAPI HTTP Routes and UI endpoints.
"""

import asyncio
import os
import sys
import httpx

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.database import init_db


async def test_routes():
    print("=" * 60)
    print("🌐 TESTING FASTAPI HTTP ENDPOINTS & WEB UI")
    print("=" * 60)

    await init_db()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Test 1: Dashboard page
        res = await client.get("/")
        assert res.status_code == 200, f"Dashboard failed with {res.status_code}"
        assert "Meeting Mind" in res.text
        print("✅ GET / (Dashboard) -> 200 OK")

        # Test 2: Upload page
        res = await client.get("/upload")
        assert res.status_code == 200
        assert "Ingest New Transcript" in res.text
        print("✅ GET /upload -> 200 OK")

        # Test 3: Confirmations queue
        res = await client.get("/confirmations")
        assert res.status_code == 200
        assert "Confirmation & Review Queue" in res.text
        print("✅ GET /confirmations -> 200 OK")

        # Test 4: Conflicts page
        res = await client.get("/conflicts")
        assert res.status_code == 200
        assert "Cross-Meeting Decision Governance" in res.text
        print("✅ GET /conflicts -> 200 OK")

        # Test 5: Team page
        res = await client.get("/team")
        assert res.status_code == 200
        assert "Registered Team Members" in res.text
        print("✅ GET /team -> 200 OK")

        # Test 6: Digests page
        res = await client.get("/digests")
        assert res.status_code == 200
        assert "Weekly Individual Intelligence Digests" in res.text
        print("✅ GET /digests -> 200 OK")

        # Test 7: API stats endpoint
        res = await client.get("/api/dashboard/stats")
        assert res.status_code == 200
        stats = res.json()
        print(f"✅ GET /api/dashboard/stats -> 200 OK (Stats: {stats})")

        # Test 8: Upload meeting via API
        sample_transcript = """
Alice: Welcome everyone. We need to decide on our CI/CD pipeline today.
Bob: Let's adopt GitHub Actions for all our deployments.
Alice: Approved! Let's use GitHub Actions. Bob, please configure the release workflow by Friday.
"""
        res = await client.post("/api/meetings/upload", json={
            "title": "CI/CD Pipeline Architecture",
            "transcript": sample_transcript,
            "format": "plain"
        })
        assert res.status_code in (200, 201)
        upload_data = res.json()
        meeting_id = upload_data["id"]
        print(f"✅ POST /api/meetings/upload -> 200 OK (Meeting ID: {meeting_id})")

        # Give background processing a second to complete
        await asyncio.sleep(2)

        # Check status
        res = await client.get(f"/api/meetings/{meeting_id}/status")
        assert res.status_code == 200
        print(f"✅ GET /api/meetings/{meeting_id}/status -> {res.json()['status']}")

        # Check detail
        res = await client.get(f"/meetings/{meeting_id}")
        assert res.status_code == 200
        print(f"✅ GET /meetings/{meeting_id} -> 200 OK")

    print("=" * 60)
    print("🎉 ALL HTTP ROUTES AND INTERACTIVE UI TEMPLATES VERIFIED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_routes())
