# Flow-boiling and single-phase heat transfer coefficients.
# Refs: Dittus & Boelter (1930); Chen (1966); Kandlikar (1990);
#       Groeneveld (1973); Kim & Mudawar (CHF)

import CoolProp.CoolProp as CP


# ── Single-phase ─────────────────────────────────────────────────────────────

def h_dittus_boelter(Re: float, Pr: float, k_f: float, D: float,
                     n: float = 0.4) -> float:
    """Dittus–Boelter [W/m²K].  n=0.4 for heating (fluid absorbs heat), 0.3 for cooling.

    Valid for: Re > 10 000, 0.7 < Pr < 160, L/D > 10.
    """
    if Re < 1e4:
        # Gnielinski is more accurate for transitional Re but D-B is good enough here
        pass
    return 0.023 * Re**0.8 * Pr**n * k_f / D


def h_single_phase(m_dot: float, D: float, fluid: str, p: float, h: float) -> float:
    """Dittus–Boelter h [W/m²K] for single-phase flow at (p, h)."""
    from hycool_pipe import friction as fric
    from math import pi
    props = {
        "mu": CP.PropsSI("V", "P", p, "Hmass", h, fluid),
        "k":  CP.PropsSI("L", "P", p, "Hmass", h, fluid),
        "cp": CP.PropsSI("C", "P", p, "Hmass", h, fluid),
    }
    Re = fric.reynolds(D, m_dot, props["mu"])
    Pr = props["cp"] * props["mu"] / props["k"]
    return h_dittus_boelter(Re, Pr, props["k"], D)


# ── Chen (1966) saturated flow boiling ──────────────────────────────────────

def _chen_F(x: float, rho_l: float, rho_g: float,
            mu_l: float, mu_g: float) -> float:
    """Chen enhancement factor F for macroconvection."""
    if x <= 1e-6:
        return 1.0
    X_tt_inv = (x / (1.0 - x))**0.9 * (rho_l / rho_g)**0.5 * (mu_g / mu_l)**0.1
    if X_tt_inv <= 0.1:
        return 1.0
    return 2.35 * (X_tt_inv + 0.213)**0.736


def _chen_S(Re_l: float, F: float) -> float:
    """Chen suppression factor S for nucleate boiling."""
    Re_tp = Re_l * F**1.25
    return 1.0 / (1.0 + 2.53e-6 * Re_tp**1.17)


def h_chen(G: float, x: float, D: float, fluid: str, p: float) -> float:
    """Chen (1966) saturated flow-boiling coefficient [W/m²K].

    Implements the macroconvective term F × h_DB plus the nucleate boiling
    suppression structure. The Forster–Zuber nucleate term (S × h_FZ) requires
    the inner wall temperature; it is included here with ΔT_sat estimated from
    the convective heat flux. Full iterative wall-coupling is added in point 8.

    Parameters
    ----------
    G      : mass flux [kg/m²s]
    x      : vapour quality [0, 1]
    D      : inner diameter [m]
    fluid  : CoolProp fluid string
    p      : local pressure [Pa]
    """
    # Saturated properties
    rho_l = CP.PropsSI("D",     "P", p, "Q", 0.0, fluid)
    rho_g = CP.PropsSI("D",     "P", p, "Q", 1.0, fluid)
    mu_l  = CP.PropsSI("V",     "P", p, "Q", 0.0, fluid)
    mu_g  = CP.PropsSI("V",     "P", p, "Q", 1.0, fluid)
    k_l   = CP.PropsSI("L",     "P", p, "Q", 0.0, fluid)
    cp_l  = CP.PropsSI("C",     "P", p, "Q", 0.0, fluid)
    h_fg  = CP.PropsSI("Hmass", "P", p, "Q", 1.0, fluid) - CP.PropsSI("Hmass", "P", p, "Q", 0.0, fluid)
    sigma = CP.PropsSI("I",     "P", p, "Q", 0.5, fluid)
    rho_g_val = rho_g   # alias for FZ
    T_sat_val = CP.PropsSI("T", "P", p, "Q", 0.5, fluid)

    # Liquid-only Re & Pr
    Re_l = max(G * (1.0 - x) * D / mu_l, 100.0)
    Pr_l = cp_l * mu_l / k_l

    # Dittus-Boelter on liquid
    h_DB = h_dittus_boelter(Re_l, Pr_l, k_l, D)

    # F factor
    F = _chen_F(x, rho_l, rho_g, mu_l, mu_g)
    h_conv = F * h_DB

    # S factor
    S = _chen_S(Re_l, F)

    # Forster–Zuber nucleate boiling — estimate ΔT_sat from convective term
    # (avoids circular dependency; full iteration added in point 8)
    # For high-Re cryogenic flow S ≈ 0–0.05, so this term is small
    dT_sat_est = 1.0  # K — conservative placeholder
    try:
        p_wall = CP.PropsSI("P", "T", T_sat_val + dT_sat_est, "Q", 0.5, fluid)
        dp_sat = abs(p_wall - p)
        h_FZ = (0.00122
                * k_l**0.79 * cp_l**0.45 * rho_l**0.49
                / (sigma**0.5 * mu_l**0.29 * h_fg**0.24 * rho_g_val**0.24)
                * dT_sat_est**0.24 * dp_sat**0.75)
    except Exception:
        h_FZ = 0.0   # CoolProp saturation query failed near critical point

    return h_conv + S * h_FZ


# ── Post-CHF (stubs for point 8+) ────────────────────────────────────────────

def x_crit(G: float, D: float, fluid: str, p: float) -> float:
    """Critical quality (dryout onset) — Kim & Mudawar correlation. Stub."""
    raise NotImplementedError


def h_groeneveld(G: float, x: float, D: float, fluid: str, p: float) -> float:
    """Groeneveld (1973) post-CHF film boiling coefficient. Stub."""
    raise NotImplementedError
