"""
Integration tests for FastAPI endpoints using TestClient.
Asserts status codes, response schemas, and strict 422 leakage prevention on G3.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    """Provides TestClient with startup lifespan executed."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_student_payload():
    """Sample student payload adhering strictly to feature schema without G3."""
    return {
        "student_id": "TEST_001",
        "sex": "F",
        "school": "GP",
        "age": 16,
        "Medu": 3,
        "Fedu": 2,
        "traveltime": 1,
        "studytime": 2,
        "failures": 1,
        "famrel": 4,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 2,
        "health": 4,
        "absences": 6,
        "G1": 8,
        "G2": 7,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Mjob": "services",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no"
    }


def test_health_endpoint(client):
    """Verifies /health returns 200 and reports model_loaded: true."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_overview_endpoint(client):
    """Verifies /overview returns expected summary metrics."""
    response = client.get("/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["total_students"] == 99
    assert data["capacity_fraction"] == 0.20
    assert data["num_selected"] == 20
    assert "worst_group_recall" in data
    assert "fairness_gap" in data
    assert "brier_score" in data


def test_students_endpoint(client):
    """Verifies /students returns full ranked validation list and handles query filters."""
    # Full list
    response = client.get("/students")
    assert response.status_code == 200
    students = response.json()
    assert len(students) == 99
    assert students[0]["rank"] == 1
    assert students[0]["need_score"] >= students[-1]["need_score"]
    
    # Filter selected_only
    resp_selected = client.get("/students?selected_only=true")
    assert resp_selected.status_code == 200
    sel_students = resp_selected.json()
    assert len(sel_students) == 20
    for s in sel_students:
        assert s["selected"] is True

    # Filter group
    resp_grp = client.get("/students?group=F")
    assert resp_grp.status_code == 200
    f_students = resp_grp.json()
    for s in f_students:
        assert s["sex"] == "F"


def test_student_detail_endpoint(client):
    """Verifies /students/{id} for both existing and non-existent IDs."""
    # Existing student
    response = client.get("/students/VAL0000")
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "VAL0000"
    assert "top_factors" in data
    assert "raw_features" in data
    assert "disclaimer" in data
    
    # Non-existent student
    resp_missing = client.get("/students/NONEXISTENT_999")
    assert resp_missing.status_code == 404
    assert "not found" in resp_missing.json()["detail"].lower()


def test_fairness_endpoint(client):
    """Verifies /fairness endpoint schema, groups, and gap metrics."""
    response = client.get("/fairness")
    assert response.status_code == 200
    data = response.json()
    assert "groups" in data
    assert "worst_group_recall" in data
    assert "fairness_gap" in data
    assert len(data["groups"]) >= 4  # F, M, GP, MS
    
    for g in data["groups"]:
        assert "eligible" in g
        assert "status_label" in g
        assert "recall" in g


def test_model_performance_endpoint(client):
    """Verifies /model-performance returns robustness benchmark and context."""
    response = client.get("/model-performance")
    assert response.status_code == 200
    data = response.json()
    assert "overview" in data
    assert "robustness" in data
    assert data["robustness"]["seeds_evaluated"] == 10
    assert "brier_score" in data["robustness"]


def test_predict_single_student_success(client, valid_student_payload):
    """Verifies POST /predict successfully scores and explains a valid student."""
    response = client.post("/predict", json=valid_student_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    pred = data["predictions"][0]
    assert 0.0 <= pred["probability"] <= 1.0
    assert 0.0 <= pred["need_score"] <= 100.0
    assert len(pred["top_factors"]) >= 1
    assert "disclaimer" in pred


def test_predict_batch_students_success(client, valid_student_payload):
    """Verifies POST /predict handles batch input."""
    student_2 = valid_student_payload.copy()
    student_2["student_id"] = "TEST_002"
    student_2["G1"] = 15
    student_2["G2"] = 16
    student_2["failures"] = 0
    
    payload = {"students": [valid_student_payload, student_2]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    # High-performing student should have lower need score
    assert data["predictions"][1]["need_score"] < data["predictions"][0]["need_score"]


def test_predict_rejects_g3_leakage_with_422(client, valid_student_payload):
    """
    CRITICAL REQUIREMENT:
    Any payload containing G3 must be rejected with HTTP 422 and a clear message,
    never accepted, never silently ignored, never 500.
    """
    leaked_payload = valid_student_payload.copy()
    leaked_payload["G3"] = 6  # Deliberate leakage
    
    response = client.post("/predict", json=leaked_payload)
    assert response.status_code == 422
    data = response.json()
    assert "leakage violation" in str(data).lower()
    assert "g3" in str(data).lower()


def test_predict_rejects_support_needed_leakage_with_422(client, valid_student_payload):
    """Verifies POST /predict rejects payloads containing support_needed with HTTP 422."""
    leaked_payload = valid_student_payload.copy()
    leaked_payload["support_needed"] = 1
    
    response = client.post("/predict", json=leaked_payload)
    assert response.status_code == 422
    data = response.json()
    assert "leakage violation" in str(data).lower()


def test_predict_rejects_out_of_range_features_with_422(client, valid_student_payload):
    """Verifies POST /predict rejects out-of-range feature values with HTTP 422."""
    invalid_payload = valid_student_payload.copy()
    invalid_payload["age"] = 55  # Out of range [15, 22]
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
