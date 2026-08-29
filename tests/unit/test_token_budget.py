import pytest

from backend.app.context_engine.token_budget import TokenBudget


def test_percentages_must_sum_to_one():
    with pytest.raises(ValueError):
        TokenBudget(0.5, 0.5, memory_pct=0.5, rag_pct=0.0, query_pct=0.0)


def test_default_percentages_are_valid():
    TokenBudget()


def test_buffer_is_reserved_before_rag_can_consume_it():
    """Regression test for the original sketch's bug: buffer_budget was computed but
    never subtracted from `remaining`, so an oversized rag_tokens request could eat
    the entire reserved margin. With a 1000-token budget and a 250-token (25%)
    buffer, RAG must never be allocated more than 1000 - system - query - buffer,
    even when it asks for far more than that.
    """
    budget = TokenBudget(total_budget=1000)
    allocations = budget.allocate(
        system_tokens=100, memory_tokens=0, rag_tokens=100000, query_tokens=100
    )

    assert allocations["buffer"] == 250
    assert allocations["rag"] <= 1000 - allocations["system"] - allocations["query"] - 250


def test_fixed_allocations_are_never_truncated_below_actual_size_within_their_cap():
    budget = TokenBudget(total_budget=1000)
    allocations = budget.allocate(
        system_tokens=50, memory_tokens=0, rag_tokens=0, query_tokens=50
    )

    assert allocations["system"] == 50
    assert allocations["query"] == 50


def test_small_requests_are_not_over_allocated():
    """Allocation reflects what was actually asked for, not the full budget cap."""
    budget = TokenBudget(total_budget=128000)
    allocations = budget.allocate(
        system_tokens=200, memory_tokens=100, rag_tokens=300, query_tokens=50
    )

    assert allocations["system"] == 200
    assert allocations["memory"] == 100
    assert allocations["rag"] == 300
    assert allocations["query"] == 50


def test_allocations_never_exceed_total_budget():
    budget = TokenBudget(total_budget=1000)
    allocations = budget.allocate(
        system_tokens=100000,
        memory_tokens=100000,
        rag_tokens=100000,
        query_tokens=100000,
    )

    assert sum(allocations.values()) <= budget.total_budget
