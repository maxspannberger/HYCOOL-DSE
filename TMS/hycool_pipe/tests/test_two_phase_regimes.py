# Validation: regime classification, flow pattern map, choking sentinel (spec §6 items 3–4)

import pytest
from hycool_pipe.regimes import classify_regime, two_phase_pattern, FlowRegime, FlowPattern
from hycool_pipe.solver import march, ChokedFlow
from hycool_pipe.cases.lh2 import LH2Config


# ── Regime classification ─────────────────────────────────────────────────────

def test_subcooled_classification():
    assert classify_regime(-1.0) == FlowRegime.SUBCOOLED_LIQUID
    assert classify_regime(-0.5) == FlowRegime.SUBCOOLED_LIQUID


def test_two_phase_classification():
    assert classify_regime(0.0)  == FlowRegime.SATURATED_TWO_PHASE
    assert classify_regime(0.5)  == FlowRegime.SATURATED_TWO_PHASE
    assert classify_regime(1.0)  == FlowRegime.SATURATED_TWO_PHASE


def test_superheated_classification():
    assert classify_regime(2.0) == FlowRegime.SUPERHEATED_VAPOUR
    assert classify_regime(1.1) == FlowRegime.SUPERHEATED_VAPOUR


# ── Mandhane flow-pattern map ─────────────────────────────────────────────────

def test_high_gas_velocity_gives_annular():
    """High j_G (> 10 m/s) → annular pattern per Mandhane (1974)."""
    # j_G = x*G/rho_g; pick values that give j_G >> 10
    pattern = two_phase_pattern(G=500.0, x=0.9, rho_l=70.0, rho_g=1.5, D=0.015)
    assert pattern == FlowPattern.ANNULAR


def test_high_liquid_velocity_gives_bubbly():
    """High j_L (> 3 m/s) and low j_G (< 0.5 m/s) → bubbly pattern.

    With G=300, x=0.001, rho_l=70, rho_g=1.5:
        j_L = 0.999*300/70 ≈ 4.28 m/s  (> 3 ✓)
        j_G = 0.001*300/1.5 = 0.20 m/s (< 0.5 ✓)
    """
    pattern = two_phase_pattern(G=300.0, x=0.001, rho_l=70.0, rho_g=1.5, D=0.015)
    assert pattern == FlowPattern.BUBBLY


def test_low_both_gives_stratified():
    """Low j_L and low j_G → stratified (common for LH2)."""
    pattern = two_phase_pattern(G=5.0, x=0.5, rho_l=70.0, rho_g=1.5, D=0.015)
    assert pattern == FlowPattern.STRATIFIED


# ── Choking sentinel (spec §6 item 4) ────────────────────────────────────────

def test_choked_flow_raised_for_tiny_pipe():
    """A high mass flow through a very small bore must raise ChokedFlow.

    At p=2 bar, T=20 K: rho_LH2 ≈ 71 kg/m³, a ≈ 1250 m/s.
    With m_dot=1 kg/s and D=2 mm:
        u = 1.0 / (71 × π×(0.002)²/4) ≈ 4500 m/s  >>  a → M >> 1.
    """
    cfg = LH2Config(m_dot=1.0)     # 14× normal flow rate
    with pytest.raises(ChokedFlow):
        march(D=0.002, t_foam=0.02, cfg=cfg, n_seg=10)


def test_choked_flow_not_raised_for_normal_conditions():
    """Normal LH2 operating conditions must NOT raise ChokedFlow."""
    cfg = LH2Config()              # default 71 g/s
    result = march(D=0.014, t_foam=0.021, cfg=cfg, n_seg=50)
    # If we get here without exception, the test passes.
    assert result.dp_total > 0
