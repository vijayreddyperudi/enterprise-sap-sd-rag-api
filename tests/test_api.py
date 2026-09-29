from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app, ChatRequest
from app.services.ingestion_service import resolve_pdf_path


client = TestClient(app)


def test_health_endpoint():
    """Verify that the /health endpoint responds with status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "index_ready" in data


def test_chat_validation_empty_query():
    """Verify that empty or whitespace-only queries are rejected with 422."""
    response = client.post("/api/v1/chat", json={"question": ""})
    assert response.status_code == 422

    response_space = client.post("/api/v1/chat", json={"question": "   "})
    assert response_space.status_code == 422


def test_chat_validation_oversized_query():
    """Verify that queries exceeding max_length (2000 chars) are rejected with 422."""
    long_question = "A" * 2001
    response = client.post("/api/v1/chat", json={"question": long_question})
    assert response.status_code == 422


def test_chat_successful_mock_response():
    """Verify that a valid question returns structured answer and sources."""
    mock_rag = MagicMock()
    mock_rag.answer.return_value = {
        "answer": "SAP SD manages sales orders, deliveries, and billing.",
        "sources": [1, 2],
        "model": "llama-3.3-70b-versatile",
    }

    with patch("app.main.rag_service", mock_rag):
        response = client.post(
            "/api/v1/chat",
            json={"question": "What is SAP SD?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "SAP SD manages sales orders, deliveries, and billing."
        assert data["sources"] == [1, 2]
        assert data["model"] == "llama-3.3-70b-versatile"


def test_resolve_pdf_path():
    """Verify that resolve_pdf_path correctly finds the sample PDF."""
    resolved = resolve_pdf_path("data/Sample.pdf")
    assert "Sample.pdf" in resolved or "sample.pdf" in resolved
