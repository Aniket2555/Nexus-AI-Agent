from dataclasses import dataclass


@dataclass
class TokenBudget:
    """Priority-based token allocation across a chat turn's prompt sections.

    `buffer_pct` is a reserved safety margin — headroom for tokenizer estimation
    error and the model's own reply. The original sketch computed `buffer_budget`
    as a property but never subtracted it from `remaining` in `allocate()`, so `rag`
    could consume the entire buffer and leave no margin at all; fixed below.
    """

    total_budget: int = 128_000  # matches Groq's llama-3.3-70b-versatile context window

    system_pct: float = 0.10
    memory_pct: float = 0.15
    rag_pct: float = 0.40
    query_pct: float = 0.10
    buffer_pct: float = 0.25

    def __post_init__(self) -> None:
        total_pct = (
            self.system_pct + self.memory_pct + self.rag_pct + self.query_pct + self.buffer_pct
        )
        if abs(total_pct - 1.0) > 1e-6:
            raise ValueError(f"Allocation percentages must sum to 1.0, got {total_pct}")

    @property
    def system_budget(self) -> int:
        return int(self.total_budget * self.system_pct)

    @property
    def memory_budget(self) -> int:
        return int(self.total_budget * self.memory_pct)

    @property
    def rag_budget(self) -> int:
        return int(self.total_budget * self.rag_pct)

    @property
    def query_budget(self) -> int:
        return int(self.total_budget * self.query_pct)

    @property
    def buffer_budget(self) -> int:
        return int(self.total_budget * self.buffer_pct)

    def allocate(
        self, system_tokens: int, memory_tokens: int, rag_tokens: int, query_tokens: int
    ) -> dict[str, int]:
        allocations: dict[str, int] = {}
        remaining = self.total_budget

        allocations["system"] = min(system_tokens, self.system_budget)
        allocations["query"] = min(query_tokens, self.query_budget)
        remaining -= allocations["system"] + allocations["query"]

        # Reserved before memory/RAG get a chance to spend it — this is the fix.
        allocations["buffer"] = min(self.buffer_budget, remaining)
        remaining -= allocations["buffer"]

        allocations["memory"] = min(memory_tokens, self.memory_budget, max(remaining, 0))
        remaining -= allocations["memory"]

        allocations["rag"] = min(rag_tokens, max(remaining, 0))
        return allocations
