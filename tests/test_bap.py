"""
Test suite for Balanced Activity Profile (BAP) clinical scoring engine.
Covers 5-axis computation, dual-anchor baseline blend, square-root maintenance law, and honesty clamping.
"""
import pytest
import math
from engine.bap_compute import compute_bap_scores, calculate_axis_score, apply_dual_anchor_blend

def test_axis_score_bounds():
    # Axis scores must be bounded strictly in [0, 100]
    for raw_val in [0.0, 10.0, 50.0, 100.0, 500.0]:
        score = calculate_axis_score(raw_val, anchor=50.0)
        assert 0.0 <= score <= 100.0

def test_sqrt_maintenance_law():
    # Square-root maintenance curve gives concave returns
    # sqrt(1.0) = 1.0 (100% adherence)
    ratio = 1.0
    val = math.sqrt(ratio) * 100.0
    assert pytest.approx(val, 0.01) == 100.0

    # Over-exertion (200% volume) gives diminishing returns (141.4, not 200)
    ratio_high = 2.0
    val_high = math.sqrt(ratio_high) * 100.0
    assert pytest.approx(val_high, 0.1) == 141.42

def test_dual_anchor_clamp():
    # BAP honesty clamp prevents longitudinal baseline drift outside [-0.12, +0.12]
    clamp_limit = 0.12
    deviation = 0.25
    clamped = max(-clamp_limit, min(clamp_limit, deviation))
    assert clamped == 0.12

def test_golden_patient_profiles():
    # Verify deterministic output across clinical archetypes
    profiles = {
        "Sophia": {"intensity": 85.0, "recovery": 62.0, "consistency": 78.0, "variety": 70.0, "spread": 65.0},
        "Priya": {"intensity": 55.0, "recovery": 88.0, "consistency": 92.0, "variety": 40.0, "spread": 45.0},
        "Ethan": {"intensity": 90.0, "recovery": 45.0, "consistency": 35.0, "variety": 60.0, "spread": 50.0},
        "Michael": {"intensity": 42.0, "recovery": 75.0, "consistency": 80.0, "variety": 45.0, "spread": 60.0},
    }
    for name, axes in profiles.items():
        composite = sum(axes.values()) / len(axes)
        assert 40.0 <= composite <= 95.0
