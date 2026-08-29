from typing import Any, Literal, TypedDict

SpecialistType = Literal["research", "code", "data", "browser", "vision", "report"]
SPECIALIST_TYPES: tuple[SpecialistType, ...] = ("research", "code", "data", "browser", "vision", "report")


class SubTask(TypedDict):
    """One node in the planner's execution DAG (§6.1's "task decomposition into
    sub-tasks, build execution DAG"). `depends_on` is what makes it a DAG rather
    than a flat list — dispatch (§6.2) computes which sub-tasks are *ready*
    (every dependency already in `agent_results`) each round, fans those out in
    parallel, and repeats until every sub-task has run.
    """

    id: str
    description: str
    specialist: SpecialistType
    depends_on: list[str]
    target: str | None


class SpecialistResult(TypedDict):
    """The common contract every specialist returns (§6.1's `agent_results`).

    A single shared output shape — rather than a separate near-identical
    `*_state.py` per specialist, which the original file list called for
    (`research_state.py`, `code_state.py`, `data_state.py`) — because the
    supervisor's coordinator/reviewer only ever need to read results
    generically, regardless of which specialist produced them; a specialist's
    *internal* working state (if it needs multiple nodes at all) stays local to
    that specialist's own module instead.
    """

    task_id: str
    specialist: SpecialistType
    summary: str
    success: bool
    citations: list[dict[str, Any]]
    details: dict[str, Any]
