# Ambient temperature profiles and external convection coefficient.
# Sizing case: "hot_ground" (315 K) per spec §1 issue 6.
# Refs: ISA atmosphere; Churchill & Chu (1975) for natural convection.

import math

# Mission phase ambient temperatures [K]
MISSION_T = {
    "hot_ground":   315.0,   # hot-day ground / idle — worst-case sizing point
    "climb":        256.0,   # ISA ~5 000 m
    "cruise_cold":  216.0,   # ISA ~11 000 m (tropopause) — coldest cruise
}


def T_amb(mission_phase: str = "hot_ground") -> float:
    """Ambient temperature [K] for the given mission phase."""
    if mission_phase not in MISSION_T:
        raise ValueError(f"Unknown mission_phase '{mission_phase}'. "
                         f"Choose from {list(MISSION_T)}.")
    return MISSION_T[mission_phase]


# ── Air properties (Sutherland / polynomial fits, valid 200–400 K) ───────────

def _air_mu(T: float) -> float:
    """Dynamic viscosity of air [Pa·s] via Sutherland's law."""
    return 1.458e-6 * T**1.5 / (T + 110.4)


def _air_k(T: float) -> float:
    """Thermal conductivity of air [W/mK]."""
    return 0.0241 * (T / 273.0)**0.82


def _air_rho(T: float, altitude_m: float = 0.0) -> float:
    """Air density [kg/m³] — ISA pressure at altitude."""
    p_sl = 101_325.0   # Pa
    # Troposphere: T_isa = 288.15 - 0.0065*h, p ~ T^5.256
    T_isa_sl = 288.15
    T_isa    = max(T_isa_sl - 0.0065 * altitude_m, 216.65)
    p        = p_sl * (T_isa / T_isa_sl)**5.2561
    R_air    = 287.05
    return p / (R_air * T)


# ── External convection coefficient ──────────────────────────────────────────

def h_outer(altitude_m: float = 0.0,
            D_outer: float = 0.058,
            T_surface: float = 21.0,
            mission_phase: str = "hot_ground") -> float:
    """Outer convective film coefficient [W/m²K] for a horizontal cylinder.

    Uses Churchill–Chu (1975) natural convection correlation at ground;
    scales with air density at altitude (reduced buoyancy at cruise).

    Parameters
    ----------
    altitude_m   : altitude [m]
    D_outer      : outer diameter of Al jacket [m]
    T_surface    : outer surface temperature [K] (≈ T_amb - small film drop)
    mission_phase: used to get ambient temperature if not overridden
    """
    T_inf = T_amb(mission_phase)
    dT    = abs(T_inf - T_surface)

    if dT < 0.1:
        return 5.0   # minimal natural convection limit [W/m²K]

    # Air properties at film temperature
    T_film = (T_inf + T_surface) / 2.0
    mu_a   = _air_mu(T_film)
    k_a    = _air_k(T_film)
    rho_a  = _air_rho(T_film, altitude_m)
    Pr_a   = 0.71   # approximately constant for air
    beta   = 1.0 / T_film   # ideal gas

    nu_a   = mu_a / rho_a
    alpha_a = k_a / (rho_a * 1005.0)   # thermal diffusivity (cp_air ≈ 1005 J/kgK)

    Ra = 9.81 * beta * dT * D_outer**3 / (nu_a * alpha_a)

    # Churchill–Chu (1975) for isothermal horizontal cylinder (all Ra)
    factor = 0.387 * Ra**(1.0/6.0) / (1.0 + (0.559 / Pr_a)**(9.0/16.0))**(8.0/27.0)
    Nu = (0.60 + factor)**2

    return Nu * k_a / D_outer
