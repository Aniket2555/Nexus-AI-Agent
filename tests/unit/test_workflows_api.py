from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.middleware.auth import Principal
from backend.app.api.middleware.rate_limit import enforce_api_rate_limit
from backend.app.api.v1.workflows import router
from backend.app.workflows.manager import get_workflow_manager
from backend.app.workflows.models import WorkflowStatus

app = FastAPI()
app.include_router(router, prefix="/api/v1")


def _fake_principal() -> Principal:
    return Principal(tenant_id="default", user_id="test-user", api_key=None)


app.dependency_overrides[enforce_api_rate_limit] = _fake_principal

_client_cm = TestClient(app)
client = _client_cm.__enter__()


def _wait_for_terminal(workflow_id: str, timeout: float = 3.0):
    import time

    manager = get_workflow_manager()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(workflow_id)
        if record.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
            return record
        time.sleep(0.02)
    raise TimeoutError(f"workflow {workflow_id} never reached a terminal status")


def test_start_and_poll_demo_workflow_to_completion():
    response = client.post("/api/v1/workflows/demo", json={"step_count": 2, "step_seconds": 0.01})
    assert response.status_code == 202

    workflow_id = response.json()["workflow_id"]

    record = _wait_for_terminal(workflow_id)
    assert record.status == WorkflowStatus.COMPLETED

    status_response = client.get(f"/api/v1/workflows/{workflow_id}")
    body = status_response.json()
    assert body["status"] == "completed"


def test_get_unknown_workflow_is_404():
    response = client.get("/api/v1/workflows/does-not-exist")
    assert response.status_code == 404


def test_cancel_unknown_workflow_is_404():
    response = client.post("/api/v1/workflows/does-not-exist/cancel")
    assert response.status_code == 404


def test_resume_a_running_workflow_is_rejected_with_409():
    response = client.post("/api/v1/workflows/demo", json={"step_count": 3, "step_seconds": 0.3})
    workflow_id = response.json()["workflow_id"]

    resume_response = client.post(f"/api/v1/workflows/{workflow_id}/resume")

    assert resume_response.status_code == 409

    get_workflow_manager().cancel(workflow_id)
    _wait_for_terminal(workflow_id)


def test_cancel_then_resume_completes_the_workflow():
    response = client.post("/api/v1/workflows/demo", json={"step_count": 3, "step_seconds": 0.15})
    workflow_id = response.json()["workflow_id"]

    import time

    time.sleep(0.2)

    cancel_response = client.post(f"/api/v1/workflows/{workflow_id}/cancel")
    assert cancel_response.status_code == 200

    _wait_for_terminal(workflow_id)

    assert client.get(f"/api/v1/workflows/{workflow_id}").json()["status"] == "cancelled"

    resume_response = client.post(f"/api/v1/workflows/{workflow_id}/resume")
    assert resume_response.status_code == 200

    record = _wait_for_terminal(workflow_id, timeout=3.0)
    assert record.status == WorkflowStatus.COMPLETED


def test_stream_yields_sse_events_ending_in_a_terminal_status():
    response = client.post("/api/v1/workflows/demo", json={"step_count": 2, "step_seconds": 0.05})
    workflow_id = response.json()["workflow_id"]

    with client.stream("GET", f"/api/v1/workflows/{workflow_id}/stream") as stream:
        assert stream.status_code == 200
        events = [line for line in stream.iter_lines() if line.startswith("data:")]

    assert events
    assert '"status": "completed"' in events[-1] or '"status":"completed"' in events[-1]
