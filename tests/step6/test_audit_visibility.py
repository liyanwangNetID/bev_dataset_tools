import numpy as np
from step6.audit_visibility import percentile


def test_percentile_is_deterministic_nearest_rank():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == 0.0
    assert percentile(values, 0.5) == 2.0
    assert percentile(values, 1.0) == 4.0


def test_percentile_empty_is_zero():
    assert percentile([], 0.5) == 0.0
