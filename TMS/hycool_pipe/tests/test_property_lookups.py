# Validation: CoolProp H2 property sanity (spec §6 item 2)
# At (p=1 bar, saturation): rho_L ≈ 70.8 kg/m³, h_fg ≈ 446 kJ/kg, T_sat ≈ 20.3 K

import CoolProp.CoolProp as CP
import pytest
from hycool_pipe import fluids


def test_h2_density_at_saturation():
    """Saturated liquid density at 1 bar should be ≈ 70.8 kg/m³ (within 2 %)."""
    rho = fluids.rho_sat_liq("hydrogen", 1e5)
    assert abs(rho - 70.8) / 70.8 < 0.02


def test_h2_latent_heat():
    """Latent heat at 1 bar should be ≈ 446 kJ/kg (within 2 %)."""
    h_fg = fluids.h_lat("hydrogen", 1e5)
    assert abs(h_fg / 1e3 - 446.0) / 446.0 < 0.02


def test_h2_saturation_temperature():
    """Saturation temperature at 1 bar should be ≈ 20.3 K (within 0.3 K)."""
    T_s = fluids.T_sat("hydrogen", 1e5)
    assert abs(T_s - 20.3) < 0.3


def test_supercritical_quality_returns_subcooled():
    """Above p_crit, quality() must return -1.0 — no two-phase dome exists."""
    p_crit = CP.PropsSI("pcrit", "hydrogen")
    p_sc = 2.0 * p_crit
    h = CP.PropsSI("Hmass", "P", p_sc, "T", 50.0, "hydrogen")
    x = fluids.quality("hydrogen", p_sc, h)
    assert x == -1.0


def test_quality_subcooled_below_dome():
    """Below the dome (liquid side), quality returns -1.0."""
    p = 1e5
    h_liq = fluids.h_sat_liq("hydrogen", p) - 1000.0   # slightly below h_l
    x = fluids.quality("hydrogen", p, h_liq)
    assert x == pytest.approx(-1.0)


def test_quality_in_two_phase_dome():
    """Inside the two-phase dome, quality returns a value in [0, 1]."""
    p = 1e5
    h_l = fluids.h_sat_liq("hydrogen", p)
    h_v = fluids.h_sat_vap("hydrogen", p)
    h_mid = 0.5 * (h_l + h_v)
    x = fluids.quality("hydrogen", p, h_mid)
    assert 0.0 <= x <= 1.0
    assert abs(x - 0.5) < 0.01
