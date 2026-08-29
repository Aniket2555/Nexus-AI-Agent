from backend.app.graphs.nodes.routing import all_tasks_complete, compute_ready_tasks

TASKS = [
    {"id": "t1", "description": "d1", "specialist": "research", "depends_on": [], "target": None},
    {"id": "t2", "description": "d2", "specialist": "code", "depends_on": ["t1"], "target": None},
    {"id": "t3", "description": "d3", "specialist": "data", "depends_on": ["t1"], "target": None},
]


def test_compute_ready_tasks_returns_tasks_with_no_dependencies_first():
    ready = compute_ready_tasks(TASKS, completed_ids=set())
    assert [t["id"] for t in ready] == ["t1"]


def test_compute_ready_tasks_unlocks_dependents_once_dependency_completes():
    ready = compute_ready_tasks(TASKS, completed_ids={"t1"})
    assert {t["id"] for t in ready} == {"t2", "t3"}


def test_compute_ready_tasks_excludes_already_completed_tasks():
    ready = compute_ready_tasks(TASKS, completed_ids={"t1", "t2", "t3"})
    assert ready == []


def test_all_tasks_complete_false_until_every_id_present():
    assert all_tasks_complete(TASKS, {"t1", "t2"}) is False
    assert all_tasks_complete(TASKS, {"t1", "t2", "t3"}) is True


def test_all_tasks_complete_true_for_empty_task_list():
    assert all_tasks_complete([], set()) is True
