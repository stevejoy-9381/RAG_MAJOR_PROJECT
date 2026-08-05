"""
tests/test_api.py — API Integration & Security Test Suite
────────────────────────────────────────────────────────────
Verifies auth security, endpoints, document validation, and status reporting.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from api import app

client = TestClient(app)


def test_health_check():
    """Verify /health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_unauthorized_access():
    """Verify protected endpoints return 401 when token is missing."""
    response = client.get("/documents")
    assert response.status_code == 401


def test_upload_unsupported_filetype():
    """Verify 400 error on uploading unsupported file extensions."""
    import time
    username = f"testuser_api_{int(time.time())}"
    reg_resp = client.post(
        "/auth/register",
        json={"username": username, "password": "password123"},
    )
    token = reg_resp.json()["access_token"]


    response = client.post(
        "/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("malicious.exe", b"binary content", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
