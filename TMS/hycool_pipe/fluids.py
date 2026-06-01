# CoolProp wrappers for cryogenic hydrogen property lookups.
# Default fluid: "hydrogen" (normal H2). Switch to "parahydrogen" for equilibrium-para feedstock.
# All state queries use (P, Hmass) inputs — valid inside the two-phase dome where (P,T) is non-unique.
# Ref: NIST REFPROP / CoolProp parahydrogen EOS

import CoolProp.CoolProp as CP

DEFAULT_FLUID = "hydrogen"

# Sentinel values for quality outside the two-phase region
_SUBCOOLED    = -1.0
_SUPERHEATED  =  2.0


def _props(output: str, fluid: str, p: float, h: float) -> float:
    return CP.PropsSI(output, "P", p, "Hmass", h, fluid)


def get_props(fluid: str, p: float, h: float) -> dict:
    """Return thermodynamic and transport properties at (p [Pa], h [J/kg]).

    Safe inside the two-phase dome — queries by (P, Hmass) throughout.

    Returns
    -------
    dict with keys:
        T       [K]       temperature
        rho     [kg/m³]   density
        mu      [Pa·s]    dynamic viscosity
        k       [W/m·K]   thermal conductivity
        cp      [J/kg·K]  isobaric specific heat
        phase   str       CoolProp phase string
        x       float     vapour quality (-1 subcooled, 0–1 two-phase, 2 superheated)
    """
    T     = _props("T",            fluid, p, h)
    rho   = _props("D",            fluid, p, h)
    mu    = _props("V",            fluid, p, h)
    k_f   = _props("L",            fluid, p, h)
    cp    = _props("C",            fluid, p, h)
    phase = CP.PhaseSI("P", p, "Hmass", h, fluid)
    x     = quality(fluid, p, h)
    return {"T": T, "rho": rho, "mu": mu, "k": k_f, "cp": cp, "phase": phase, "x": x}


def quality(fluid: str, p: float, h: float) -> float:
    """Vapour quality x at (p, h).

    Returns -1.0 if subcooled liquid, 0–1 inside the dome, 2.0 if superheated.
    Returns -1.0 for supercritical states (p > p_crit) — no saturation dome exists.
    Never queries CoolProp by (p, T) — undefined inside the dome.
    """
    p_crit = CP.PropsSI("pcrit", fluid)
    if p >= p_crit:
        return _SUBCOOLED   # supercritical — single-phase throughout

    try:
        h_l = h_sat_liq(fluid, p)
        h_v = h_sat_vap(fluid, p)
    except Exception:
        return _SUBCOOLED   # CoolProp failed at saturation boundary

    if h <= h_l:
        return _SUBCOOLED
    if h >= h_v:
        return _SUPERHEATED
    return (h - h_l) / (h_v - h_l)


def h_sat_liq(fluid: str, p: float) -> float:
    """Saturated liquid enthalpy [J/kg] at pressure p [Pa]."""
    return CP.PropsSI("Hmass", "P", p, "Q", 0.0, fluid)


def h_sat_vap(fluid: str, p: float) -> float:
    """Saturated vapour enthalpy [J/kg] at pressure p [Pa]."""
    return CP.PropsSI("Hmass", "P", p, "Q", 1.0, fluid)


def T_sat(fluid: str, p: float) -> float:
    """Saturation temperature [K] at pressure p [Pa]."""
    return CP.PropsSI("T", "P", p, "Q", 0.5, fluid)


def h_lat(fluid: str, p: float) -> float:
    """Latent heat of vaporisation [J/kg] at pressure p [Pa]."""
    return h_sat_vap(fluid, p) - h_sat_liq(fluid, p)


def sound_speed(fluid: str, p: float, h: float) -> float:
    """Single-phase speed of sound [m/s] at (p, h).

    In the two-phase region CoolProp returns the homogeneous mixture value,
    which overestimates the true two-phase speed of sound. Use
    sound_speed_two_phase() from solver.py for choking checks in two-phase flow.
    """
    return _props("A", fluid, p, h)


def rho_sat_liq(fluid: str, p: float) -> float:
    """Saturated liquid density [kg/m³] at pressure p [Pa]."""
    return CP.PropsSI("D", "P", p, "Q", 0.0, fluid)


def rho_sat_vap(fluid: str, p: float) -> float:
    """Saturated vapour density [kg/m³] at pressure p [Pa]."""
    return CP.PropsSI("D", "P", p, "Q", 1.0, fluid)


def mu_sat_liq(fluid: str, p: float) -> float:
    """Saturated liquid dynamic viscosity [Pa·s] at pressure p [Pa]."""
    return CP.PropsSI("V", "P", p, "Q", 0.0, fluid)


def mu_sat_vap(fluid: str, p: float) -> float:
    """Saturated vapour dynamic viscosity [Pa·s] at pressure p [Pa]."""
    return CP.PropsSI("V", "P", p, "Q", 1.0, fluid)


def sigma_surface(fluid: str, p: float) -> float:
    """Surface tension [N/m] at pressure p — needed for Kandlikar boiling correlation."""
    return CP.PropsSI("I", "P", p, "Q", 0.5, fluid)


# ── Validation helper ────────────────────────────────────────────────────────

def check_property_sanity(fluid: str = DEFAULT_FLUID) -> bool:
    """Quick sanity check at (p=1 bar, sat liquid).

    Expected for parahydrogen/hydrogen at 1 bar:
        rho_L ≈ 70.8 kg/m³, h_fg ≈ 446 kJ/kg, T_sat ≈ 20.3 K
    Prints results and returns True if within 5 % of expected values.
    """
    p = 1.0e5
    rho_l  = rho_sat_liq(fluid, p)
    h_fg   = h_lat(fluid, p)
    T_s    = T_sat(fluid, p)

    checks = [
        ("rho_L [kg/m³]",  rho_l,  70.8,  5.0),
        ("h_fg [kJ/kg]",   h_fg / 1000.0, 446.0, 5.0),
        ("T_sat [K]",      T_s,    20.3,  2.0),
    ]

    print(f"\n=== CoolProp sanity check: {fluid} at p=1 bar ===")
    print(f"  {'Quantity':<20} {'Computed':>12} {'Expected':>12} {'Error %':>9}  Status")
    print("  " + "-" * 60)
    all_pass = True
    for label, computed, expected, tol in checks:
        err = abs(computed - expected) / expected * 100.0
        ok  = err <= tol
        if not ok:
            all_pass = False
        print(f"  {label:<20} {computed:>12.4f} {expected:>12.4f} {err:>8.2f}%  {'PASS' if ok else 'FAIL'}")
    print("  " + "-" * 60)
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass
