from backend.app.graphs.states.shared_state import SubTask


def compute_ready_tasks(sub_tasks: list[SubTask], completed_ids: set[str]) -> list[SubTask]:
    """The DAG's ready frontier for this dispatch round: not yet run, and every
    dependency already in `completed_ids`.

    Dispatch (§6.2) is a single node that runs every ready task concurrently
    (`asyncio.gather`) and loops the graph back to itself — via `all_tasks_complete`
    below as the conditional edge — until nothing is left, rather than LangGraph's
    `Send` API (used for Phase 3's self-consistency fan-out). `Send` fits a known,
    fixed-arity fan-out; a dependency-respecting scheduler needs multiple rounds
    whose size depends on results from the previous round, which is simpler to
    express as an explicit Python loop across graph re-entries than as nested Send
    dispatches.
    """
    return [
        task
        for task in sub_tasks
        if task["id"] not in completed_ids
        and all(dep in completed_ids for dep in task["depends_on"])
    ]


def all_tasks_complete(sub_tasks: list[SubTask], completed_ids: set[str]) -> bool:
    return all(task["id"] in completed_ids for task in sub_tasks)
