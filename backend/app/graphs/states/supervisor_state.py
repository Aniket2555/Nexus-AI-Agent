import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from backend.app.graphs.states.shared_state import SpecialistResult, SubTask


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    current_plan: str | None
    sub_tasks: list[SubTask]

    agent_results: Annotated[list[SpecialistResult], operator.add]

    draft_response: str | None
    quality_score: float
    reflection_feedback: str
    iteration_count: int

    cost_usd_spent: float

    review_passed: bool | None
    review_feedback: str

    final_response: str | None
