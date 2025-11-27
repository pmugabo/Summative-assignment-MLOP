import pytest
import os

def test_health_check(client):
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint(client, tmp_path):
    # Create a dummy audio file
    test_file = tmp_path / "test.wav"
    test_file.write_bytes(b"dummy audio data")
    
    with open(test_file, "rb") as f:
        response = client.post(
            "/predict/",
            files={"file": ("test.wav", f, "audio/wav")}
        )
    
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_metrics_endpoint(client):
    response = client.get("/metrics/")
    assert response.status_code == 200
    assert "http_request_duration_seconds" in response.text