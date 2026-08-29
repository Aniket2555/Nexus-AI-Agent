import ast
import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from backend.app.graphs.states.shared_state import SpecialistResult
from backend.app.llm.provider import get_chat_model
from backend.app.mcp.tools.custom.python_repl import run_python
from backend.app.utils.messages import as_text

CODE_PROMPT = (
    "You are the Code specialist in a multi-agent system.\n\n"
    "Write code that satisfies the task below. Output a single fenced code block with\n"
    "the language tag set correctly (e.g. ```python). Briefly explain your approach\n"
    "before the code block. State any assumptions about inputs/environment explicitly.\n\n"
    "Task: {task}\n"
)

_CODE_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


class CodeState(TypedDict):
    task: str
    response_text: str
    language: str
    code: str
    syntax_valid: bool
    syntax_error: str | None
    executed: bool
    execution_stdout: str
    execution_stderr: str
    execution_success: bool
    execution_timed_out: bool
    execution_exit_code: int | None
    execution_truncated: bool


async def generate_node(state: CodeState) -> dict[str, Any]:
    response = await get_chat_model().ainvoke(
        [SystemMessage(content=CODE_PROMPT.format(task=state["task"]))]
    )
    return {"response_text": as_text(response)}


def _extract_code_block(text: str) -> tuple[str, str]:
    match = _CODE_FENCE.search(text)
    if not match:
        return "", ""
    language, code = match.group(1) or "", match.group(2)
    return language.strip().lower(), code.strip()


async def validate_node(state: CodeState) -> dict[str, Any]:
    """Static validation, ahead of execution.

    `ast.parse()` (Python only — the one language this project can validate
    without a runtime) catches syntax errors before anything is ever run, so a
    request that can't parse doesn't cost a container spin-up.  Runtime
    execution (`execute_node`, below) exists as of Phase 7's sandbox; before
    that, this node's output was the final answer and the module docstring
    said so — kept here as the "can this even be parsed" gate feeding
    `validate_edge`, not because execution still doesn't exist.
    """
    language, code = _extract_code_block(state["response_text"])
    if not code:
        return {
            "language": language,
            "code": "",
            "syntax_valid": False,
            "syntax_error": "No code block found in the model's response.",
        }

    if language not in ("python", "py", ""):
        return {"language": language, "code": code, "syntax_valid": True, "syntax_error": None}

    try:
        ast.parse(code)
        return {
            "language": language or "python",
            "code": code,
            "syntax_valid": True,
            "syntax_error": None,
        }
    except SyntaxError as exc:
        return {
            "language": language or "python",
            "code": code,
            "syntax_valid": False,
            "syntax_error": f"{exc.msg} (line {exc.lineno})",
        }


async def execute_node(state: CodeState) -> dict[str, Any]:
    """Actually run syntax-valid Python in §7.1's Docker sandbox — the step
    Phase 5 and Phase 6 both deferred ("two phases of arbitrary code execution
    on the host" being exactly the mistake the original plan made) because
    nothing existed yet to run it safely in. It does now.
    """
    result = await run_python(state["code"])
    return {
        "executed": True,
        "execution_stdout": result["stdout"],
        "execution_stderr": result["stderr"],
        "execution_success": result["success"],
        "execution_timed_out": result["timed_out"],
        "execution_exit_code": result["exit_code"],
        "execution_truncated": result["truncated"],
    }


def validate_edge(state: CodeState) -> Literal["execute", "skip"]:
    language = state.get("language", "")
    if language in ("python", "py", "") and state.get("syntax_valid"):
        return "execute"
    return "skip"


builder = StateGraph(CodeState)
builder.add_node("generate", generate_node)
builder.add_node("validate", validate_node)
builder.add_node("execute", execute_node)
builder.set_entry_point("generate")
builder.add_edge("generate", "validate")
builder.add_conditional_edges("validate", validate_edge, {"execute": "execute", "skip": END})
builder.add_edge("execute", END)
code_graph = builder.compile()


async def run_code(task_id: str, task: str) -> SpecialistResult:
    initial_state: CodeState = {
        "task": task,
        "response_text": "",
        "language": "",
        "code": "",
        "syntax_valid": False,
        "syntax_error": None,
        "executed": False,
        "execution_stdout": "",
        "execution_stderr": "",
        "execution_success": False,
        "execution_timed_out": False,
        "execution_exit_code": None,
        "execution_truncated": False,
    }
    result = await code_graph.ainvoke(initial_state)

    executed = result.get("executed", False)
    if executed:
        success = result.get("execution_success", False)
        note = "Executed in the Phase 7 sandbox."
    elif result.get("language", "") in ("python", "py", ""):
        success = bool(result.get("syntax_valid"))
        note = "Syntax-invalid — not executed."
    else:
        success = bool(result.get("syntax_valid"))
        note = "Non-Python code is not executed, only lightly checked for a code block."

    return SpecialistResult(
        task_id=task_id,
        specialist="code",
        summary=result.get("response_text", ""),
        success=success,
        citations=[],
        details={
            "language": result.get("language", ""),
            "code": result.get("code", ""),
            "syntax_valid": result.get("syntax_valid", False),
            "syntax_error": result.get("syntax_error"),
            "executed": executed,
            "execution_stdout": result.get("execution_stdout", ""),
            "execution_stderr": result.get("execution_stderr", ""),
            "execution_timed_out": result.get("execution_timed_out", False),
            "execution_exit_code": result.get("execution_exit_code"),
            "execution_truncated": result.get("execution_truncated", False),
            "note": note,
        },
    )
