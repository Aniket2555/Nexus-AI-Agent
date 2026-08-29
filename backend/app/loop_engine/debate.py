from typing import Any, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from backend.app.llm.provider import get_chat_model
from backend.app.utils.messages import as_text

ADVOCATE_PROMPT = (
    "You are arguing FOR the most likely answer to this question, based only on "
    "the provided context. Be persuasive, but stay grounded — do not invent "
    "support that isn't actually in the context.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nYour argument (2-4 sentences):"
)

CRITIC_PROMPT = (
    "Critique the argument below. Identify any claims it makes that the context "
    "doesn't actually support, alternative interpretations it ignores, or "
    "caveats it omits. Be specific — cite what's missing, don't just express "
    "doubt.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\n"
    "Argument to critique:\n{advocate_argument}\n\nYour critique (2-4 sentences):"
)

MODERATOR_PROMPT = (
    "Weigh the argument and the critique below, then give a final, balanced "
    "answer to the question. State your confidence, and note any caveat from "
    "the critique that still stands.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\n"
    "Argument:\n{advocate_argument}\n\nCritique:\n{critic_argument}\n\nFinal answer:"
)


class DebateState(TypedDict):
    question: str
    context: str
    advocate_argument: str
    critic_argument: str
    final_answer: str


async def advocate_node(state: DebateState) -> dict[str, Any]:
    model = get_chat_model(temperature=0.3)
    prompt = ADVOCATE_PROMPT.format(context=state["context"], question=state["question"])
    response = await model.ainvoke([SystemMessage(content=prompt)])
    return {"advocate_argument": as_text(response)}


async def critic_node(state: DebateState) -> dict[str, Any]:
    model = get_chat_model(temperature=0.3)
    prompt = CRITIC_PROMPT.format(
        context=state["context"],
        question=state["question"],
        advocate_argument=state.get("advocate_argument", ""),
    )
    response = await model.ainvoke([SystemMessage(content=prompt)])
    return {"critic_argument": as_text(response)}


async def moderator_node(state: DebateState) -> dict[str, Any]:
    model = get_chat_model(temperature=0.1)
    prompt = MODERATOR_PROMPT.format(
        context=state["context"],
        question=state["question"],
        advocate_argument=state.get("advocate_argument", ""),
        critic_argument=state.get("critic_argument", ""),
    )
    response = await model.ainvoke([SystemMessage(content=prompt)])
    return {"final_answer": as_text(response)}


builder = StateGraph(DebateState)
builder.add_node("advocate", advocate_node)
builder.add_node("critic", critic_node)
builder.add_node("moderator", moderator_node)
builder.set_entry_point("advocate")
builder.add_edge("advocate", "critic")
builder.add_edge("critic", "moderator")
builder.add_edge("moderator", END)
debate_graph = builder.compile()
