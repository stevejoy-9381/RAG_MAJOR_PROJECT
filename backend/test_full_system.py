"""
test_full_system.py — Programmatic validation of the entire RAG pipeline
"""

import os
import sys
import json
import time
import requests

os.environ["TRANSFORMERS_NO_TF"] = "1"

BASE_URL = "http://localhost:8000"

def test_system():
    print(f"\n{'='*70}")
    print("  RUNNING COMPLETE RAG SYSTEM VERIFICATION TEST")
    print(f"{'='*70}\n")

    # 1. Health check
    print("[TEST 1] Checking /health endpoint...")
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200, f"Health check failed: {res.status_code}"
    health_data = res.json()
    print(f"  -> Health OK: {health_data}")

    # 2. Register user
    test_user = f"auto_tester_{int(time.time())}"
    test_pass = "TestPass123!"
    print(f"\n[TEST 2] Registering user '{test_user}' via /auth/register...")
    reg_payload = {"username": test_user, "password": test_pass}
    res = requests.post(f"{BASE_URL}/auth/register", json=reg_payload)
    assert res.status_code == 201, f"Registration failed: {res.text}"
    token_data = res.json()
    token = token_data["access_token"]
    user_id = token_data["user_id"]
    print(f"  -> Registration successful! JWT token acquired. User ID: {user_id}")

    headers = {"Authorization": f"Bearer {token}"}

    # 3. Check status (protected)
    print("\n[TEST 3] Checking /status endpoint with JWT token...")
    res = requests.get(f"{BASE_URL}/status", headers=headers)
    assert res.status_code == 200, f"Status check failed: {res.text}"
    print(f"  -> Status OK: {res.json()}")

    # 4. Create and Upload a test document
    doc_content = (
        "Quantum Neural Engine (QNE) is an experimental hybrid AI processor created by CyberDyne Labs. "
        "It operates at superconducting temperatures of 4 Kelvin and delivers 12.8 PetaFLOPS per watt. "
        "QNE was designed specifically for real-time climate simulation models."
    )
    doc_filename = "cyberdyne_qne_specs.txt"
    with open(doc_filename, "w", encoding="utf-8") as f:
        f.write(doc_content)

    print(f"\n[TEST 4] Uploading test document '{doc_filename}' via /upload...")
    with open(doc_filename, "rb") as f:
        files = {"file": (doc_filename, f, "text/plain")}
        res = requests.post(f"{BASE_URL}/upload", headers=headers, files=files)
    assert res.status_code == 200, f"Upload failed: {res.text}"
    upload_res = res.json()
    print(f"  -> Upload successful! {upload_res}")

    # Clean up local test file
    if os.path.exists(doc_filename):
        os.remove(doc_filename)

    # 5. List documents
    print("\n[TEST 5] Verifying document library via /documents...")
    res = requests.get(f"{BASE_URL}/documents", headers=headers)
    assert res.status_code == 200
    docs_res = res.json()
    print(f"  -> Documents listed: {docs_res}")

    # 6. Ask question via /chat (non-streaming)
    question = "What is the operating temperature and power efficiency of the Quantum Neural Engine?"
    print(f"\n[TEST 6] Asking Question via /chat endpoint:")
    print(f"  Question: '{question}'")
    chat_payload = {"question": question, "llm_mode": "auto"}
    res = requests.post(f"{BASE_URL}/chat", headers=headers, json=chat_payload)
    assert res.status_code == 200, f"Chat call failed: {res.text}"
    chat_res = res.json()
    print(f"\n[AI ANSWER]:\n{'─'*60}\n{chat_res['answer']}\n{'─'*60}")
    print(f"  -> Provider: {chat_res.get('provider')}")
    print(f"  -> Sources Cited: {len(chat_res.get('sources', []))} passage(s)")

    # 7. Ask question via /stream (SSE streaming)
    print(f"\n[TEST 7] Testing SSE Streaming Answer via /stream endpoint...")
    stream_payload = {"question": "Who created the QNE and what is its primary use case?", "llm_mode": "auto"}
    res = requests.post(f"{BASE_URL}/stream", headers=headers, json=stream_payload, stream=True)
    assert res.status_code == 200, f"Stream call failed: {res.text}"

    streamed_text = ""
    for line in res.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith("data: "):
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                    if event.get("type") == "token":
                        streamed_text += event.get("content", "")
                except json.JSONDecodeError:
                    pass

    print(f"\n[STREAMED AI ANSWER]:\n{'─'*60}\n{streamed_text}\n{'─'*60}\n")
    assert "CyberDyne" in streamed_text or "climate" in streamed_text, "Streamed answer missing expected facts"

    print(f"\n{'='*70}")
    print("  ALL RAG BACKEND PIPELINE TESTS PASSED 100% SUCCESSFULLY!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    test_system()
