from backend.app.evaluation.retrieval_metrics import context_precision, context_recall


def test_precision_all_retrieved_relevant():
    assert context_precision(["a", "b"], ["a", "b", "c"]) == 1.0


def test_precision_half_retrieved_relevant():
    assert context_precision(["a", "x"], ["a", "b"]) == 0.5


def test_precision_nothing_retrieved_is_zero():
    assert context_precision([], ["a"]) == 0.0


def test_recall_all_expected_found():
    assert context_recall(["a", "b", "c"], ["a", "b"]) == 1.0


def test_recall_half_expected_found():
    assert context_recall(["a"], ["a", "b"]) == 0.5


def test_recall_with_no_expected_chunks_is_trivially_satisfied():
    assert context_recall(["a"], []) == 1.0
