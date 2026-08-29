import re

from langchain_core.language_models.chat_models import BaseChatModel

from backend.app.utils.messages import as_text

CLAIM_EXTRACTION_PROMPT = (
    "Decompose the following answer into its individual atomic factual\n"
    "claims — one claim per line, numbered. A claim is a single verifiable "
    "statement; split\ncompound sentences. If the answer makes no factual claims "
    "(e.g. it's a greeting or a\nrefusal), output nothing.\n\nAnswer:\n{answer}\n"
)

CLAIM_VERDICT_PROMPT = (
    "You are checking factual claims against a context, one at a time.\n"
    "For each numbered claim below, output one line in the exact form:\n\n"
    "<number>: SUPPORTED\nor\n<number>: UNSUPPORTED\n\n"
    "A claim is SUPPORTED only if the context directly states or clearly implies "
    "it — not if it\nmerely sounds plausible. Output nothing else, no explanation.\n\n"
    "Context:\n{context}\n\nClaims:\n{claims}\n"
)

_NUMBERED_LINE = re.compile(r"^\s*\d+[.):]?\s*(.+)$")
_VERDICT_LINE = re.compile(r"(\d+)\s*:\s*(SUPPORTED|UNSUPPORTED)", re.I)


async def extract_claims(answer: str, judge: BaseChatModel) -> list[str]:
    """Split `answer` into atomic factual claims via one LLM call.

    Claim-by-claim decomposition, not a single holistic yes/no — a response with
    nine correct claims and one fabricated one should score as ~10% hallucinated,
    not pass-or-fail as a whole. This is what `faithfulness.py`'s single-number
    `faithfulness()` judgment structurally can't distinguish from "mostly right."
    """
    response = await judge.ainvoke(CLAIM_EXTRACTION_PROMPT.format(answer=answer))

    claims = []
    for line in as_text(response).splitlines():
        match = _NUMBERED_LINE.match(line)
        text = match.group(1).strip() if match else line.strip()
        if text:
            claims.append(text)
    return claims


async def _check_claims(claims: list[str], context: str, judge: BaseChatModel) -> list[bool]:
    """Verdict per claim (`True` == supported), in one batched LLM call rather than
    one call per claim — a ten-claim answer would otherwise cost ten round trips.
    Falls back to "unsupported" for any claim whose verdict line didn't parse: an
    unparseable verdict is exactly the kind of ambiguity that should count against
    the answer, not silently drop out of the denominator.
    """
    numbered = "\n".join(f"{i + 1}. {claim}" for i, claim in enumerate(claims))
    response = await judge.ainvoke(CLAIM_VERDICT_PROMPT.format(context=context, claims=numbered))

    found = _VERDICT_LINE.findall(as_text(response))
    verdicts = {int(n): v.upper() == "SUPPORTED" for n, v in found}

    return [verdicts.get(i + 1, False) for i in range(len(claims))]


async def hallucination_rate(
    answer: str, context: str, judge: BaseChatModel | None = None
) -> float | None:
    """Fraction of `answer`'s atomic claims that are *not* supported by `context`.

    `None` (not `0.0`) with no judge, and `None` for an answer with no extractable
    claims (a greeting has no claims to hallucinate — "0% hallucinated" would be
    technically true but misleading to average into a report next to real scores).
    """
    if judge is None:
        return None

    claims = await extract_claims(answer, judge)
    if not claims:
        return None

    verdicts = await _check_claims(claims, context, judge)
    unsupported = sum(1 for supported in verdicts if not supported)
    return unsupported / len(claims)


async def groundedness(
    answer: str, context: str, judge: BaseChatModel | None = None
) -> float | None:
    """1 - hallucination_rate, exposed under the name `rag_evaluator.py`'s metric
    table (§8.5) uses. Same underlying claim-by-claim computation as
    `hallucination_rate` — not a second, cheaper holistic judgment like
    `faithfulness.py`'s `faithfulness()` is, deliberately: groundedness is meant to
    be the more rigorous of the two.
    """
    rate = await hallucination_rate(answer, context, judge)
    return None if rate is None else 1.0 - rate
