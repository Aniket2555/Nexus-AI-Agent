from langgraph.types import Command

from backend.app.loop_engine.human_approval import human_approval_graph


async def test_graph_pauses_with_the_action_details_in_the_interrupt_payload():
    config = {"configurable": {"thread_id": "test-approve"}}

    result = await human_approval_graph.ainvoke(
        {
            "pending_action": {"tool": "send_email", "to": "customer@example.com"},
            "impact_assessment": "external communication",
        },
        config,
    )

    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1

    payload = interrupts[0].value
    assert payload["action"] == {"tool": "send_email", "to": "customer@example.com"}
    assert payload["estimated_impact"] == "external communication"
    assert "human_approved" not in result


async def test_resuming_with_approval_continues_the_run():
    config = {"configurable": {"thread_id": "test-resume-approved"}}

    await human_approval_graph.ainvoke(
        {"pending_action": {"tool": "delete_file"}, "impact_assessment": "destructive"}, config
    )

    result = await human_approval_graph.ainvoke(Command(resume={"approved": True}), config)

    assert result["human_approved"] is True
    assert "__interrupt__" not in result


async def test_resuming_with_denial_is_reflected_in_state():
    config = {"configurable": {"thread_id": "test-resume-denied"}}

    await human_approval_graph.ainvoke(
        {"pending_action": {"tool": "delete_file"}, "impact_assessment": "destructive"}, config
    )

    result = await human_approval_graph.ainvoke(Command(resume={"approved": False}), config)

    assert result["human_approved"] is False


async def test_different_threads_have_independent_pending_interrupts():
    """One thread's pause must not leak into another's — thread_id is the isolation
    boundary the checkpointer keys on."""
    config_a = {"configurable": {"thread_id": "thread-a"}}
    config_b = {"configurable": {"thread_id": "thread-b"}}

    await human_approval_graph.ainvoke(
        {"pending_action": {"id": "a"}, "impact_assessment": "low"}, config_a
    )
    await human_approval_graph.ainvoke(
        {"pending_action": {"id": "b"}, "impact_assessment": "low"}, config_b
    )

    resume_a = Command(resume={"approved": True})
    result_a = await human_approval_graph.ainvoke(resume_a, config_a)

    assert result_a["pending_action"]["id"] == "a"
    assert result_a["human_approved"] is True

    resume_b = Command(resume={"approved": False})
    result_b = await human_approval_graph.ainvoke(resume_b, config_b)

    assert result_b["pending_action"]["id"] == "b"
    assert result_b["human_approved"] is False
