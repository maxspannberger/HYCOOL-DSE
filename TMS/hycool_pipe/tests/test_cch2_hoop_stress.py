# Validation: CcH2 structural sizing (spec §6 item 5)
# At p=350 bar, D=5 mm, SF=2.5, gaseous H2: t_ss must be well above the 0.3 mm floor.

import pytest
from hycool_pipe.solver import size_t_ss
from hycool_pipe.materials import sigma_allow


def test_hoop_stress_gives_thick_wall_at_350_bar():
    """At 350 bar / D=5 mm / SF=2.5 / gaseous H2, t_ss should exceed 1 mm (spec ≥ 0.6 mm)."""
    t = size_t_ss(D=0.005, p_design=350e5, SF=2.5, T=50.0,
                  h2_exposure="gaseous", t_min=3e-4)
    # sigma_allow(50 K, gaseous) ≈ 130 MPa
    # t = 350e5 * 0.005 * 2.5 / (2 * 130e6) ≈ 1.68 mm
    assert t > 0.6e-3, f"Expected t_ss > 0.6 mm, got {t*1e3:.3f} mm"
    assert t > 3e-4,   "t_ss must be at least at the manufacturing floor"


def test_hoop_stress_scales_linearly_with_pressure():
    """Doubling the design pressure should double t_ss (hoop stress is linear in p)."""
    t_min_dummy = 1e-9   # negligible floor so structural sizing governs
    t_low  = size_t_ss(D=0.01, p_design=100e5, SF=2.5, T=50.0,
                       h2_exposure="gaseous", t_min=t_min_dummy)
    t_high = size_t_ss(D=0.01, p_design=200e5, SF=2.5, T=50.0,
                       h2_exposure="gaseous", t_min=t_min_dummy)
    assert t_high / t_low == pytest.approx(2.0, rel=1e-3)


def test_lh2_pressure_floor_governs():
    """At LH2 operating pressure (2 bar), hoop stress is ~2 μm — manufacturing floor governs."""
    t = size_t_ss(D=0.015, p_design=2e5, SF=2.5, T=20.0,
                  h2_exposure="liquid", t_min=3e-4)
    assert t == pytest.approx(3e-4, rel=1e-4), (
        f"Floor should govern at 2 bar; got t_ss={t*1e3:.4f} mm"
    )


def test_gaseous_tighter_than_liquid_derating():
    """Gaseous H2 de-rating gives a larger t_ss than liquid H2 for the same conditions."""
    t_gas = size_t_ss(D=0.01, p_design=350e5, SF=2.5, T=50.0,
                      h2_exposure="gaseous", t_min=1e-9)
    t_liq = size_t_ss(D=0.01, p_design=350e5, SF=2.5, T=50.0,
                      h2_exposure="liquid", t_min=1e-9)
    assert t_gas > t_liq, "Gaseous H2 de-rating must yield thicker wall than liquid"


def test_sigma_allow_cryogenic_enhancement():
    """sigma_allow should be higher at 20 K than at 300 K (cryogenic strengthening)."""
    sig_cryo = sigma_allow(20.0, h2_exposure="liquid")
    sig_warm = sigma_allow(300.0, h2_exposure="liquid")
    assert sig_cryo > sig_warm
