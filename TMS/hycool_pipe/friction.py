# Friction factor correlations: single-phase and two-phase multipliers.
# Refs: Colebrook & White (1937); Haaland (1983);
#       Müller-Steinhagen & Heck (1986); Friedel (1979); Lockhart & Martinelli (1949)

import warnings
from math import log, log10, sqrt, pi
from scipy.optimize import brentq

# SS-316L drawn tubing roughness (m)
_DEFAULT_ROUGHNESS = 1.5e-6

_RE_LAM  = 2300   # laminar–transition boundary
_RE_TURB = 4000   # transition–turbulent boundary


# ── Reynolds number helper ───────────────────────────────────────────────────

def reynolds(D: float, m_dot: float, mu: float) -> float:
    """Pipe Reynolds number Re = 4*m_dot / (pi*D*mu)."""
    return 4.0 * m_dot / (pi * D * mu)


# ── Single-phase friction factors ────────────────────────────────────────────

def colebrook(Re: float, D: float, roughness: float = _DEFAULT_ROUGHNESS) -> float:
    """Darcy friction factor via Colebrook–White (iterative).

    Covers laminar (Re < 2300), transitional (2300–4000, with warning),
    and fully turbulent (Re > 4000) regimes.

    Colebrook–White implicit form:
        1/sqrt(f) = -2 * log10(eps/(3.7*D) + 2.51/(Re*sqrt(f)))
    """
    if Re <= 0:
        raise ValueError(f"Re must be positive, got {Re:.3g}")

    if Re < _RE_LAM:
        return 64.0 / Re

    if Re < _RE_TURB:
        warnings.warn(
            f"Re={Re:.0f} is in the laminar-turbulent transition region "
            f"({_RE_LAM}–{_RE_TURB}). Friction factor is uncertain.",
            UserWarning, stacklevel=2,
        )

    eps_over_D = roughness / D

    # Haaland as starting bracket / initial guess
    f_haaland = haaland(Re, D, roughness)

    def colebrook_residual(f):
        return 1.0 / sqrt(f) + 2.0 * log10(eps_over_D / 3.7 + 2.51 / (Re * sqrt(f)))

    # Bracket: f is bounded by smooth-pipe limit and fully-rough limit
    f_lo = max(f_haaland * 0.5, 1e-6)
    f_hi = min(f_haaland * 2.0, 0.5)

    # Ensure the bracket actually brackets a root
    if colebrook_residual(f_lo) * colebrook_residual(f_hi) > 0:
        # Widen bracket
        f_lo, f_hi = 1e-6, 0.5

    return brentq(colebrook_residual, f_lo, f_hi, xtol=1e-10, rtol=1e-8)


def haaland(Re: float, D: float, roughness: float = _DEFAULT_ROUGHNESS) -> float:
    """Darcy friction factor via Haaland (1983) explicit approximation.

    Maximum error vs Colebrook–White: ~1.5 % for Re > 4000.
    Falls back to laminar formula for Re < 2300.
    """
    if Re <= 0:
        raise ValueError(f"Re must be positive, got {Re:.3g}")

    if Re < _RE_LAM:
        return 64.0 / Re

    eps_over_D = roughness / D
    inv_sqrt_f = -1.8 * log10((eps_over_D / 3.7) ** 1.11 + 6.9 / Re)
    return 1.0 / inv_sqrt_f ** 2


def friction_factor(Re: float, D: float, roughness: float = _DEFAULT_ROUGHNESS,
                    method: str = "colebrook") -> float:
    """Darcy friction factor dispatcher.

    Parameters
    ----------
    method : "colebrook" (default, iterative) or "haaland" (explicit, ~1.5% error).
    """
    if method == "colebrook":
        return colebrook(Re, D, roughness)
    if method == "haaland":
        return haaland(Re, D, roughness)
    raise ValueError(f"Unknown method '{method}'. Choose 'colebrook' or 'haaland'.")


# ── Two-phase friction multipliers (implemented in point 7) ─────────────────

def phi2_lo_msh(x: float, G: float, D: float,
                rho_l: float, rho_g: float,
                mu_l: float, mu_g: float,
                roughness: float = _DEFAULT_ROUGHNESS) -> float:
    """Müller-Steinhagen–Heck (1986) two-phase friction multiplier Φ²_LO.

    Default correlation for cryogenic horizontal flow.

    (dp/dz)_TP = [A + 2(B-A)·x]·(1-x)^(1/3) + B·x^3
    Φ²_LO = (dp/dz)_TP / (dp/dz)_LO

    where A = f_LO·G²/(2ρ_L·D)  (liquid-only pressure gradient)
          B = f_GO·G²/(2ρ_G·D)  (gas-only pressure gradient)

    Parameters
    ----------
    x         : vapour quality [–]  (clamped to [0, 1])
    G         : mass flux [kg/m²s]
    D         : inner diameter [m]
    rho_l/g   : saturated liquid/vapour density [kg/m³]
    mu_l/g    : saturated liquid/vapour dynamic viscosity [Pa·s]
    roughness : pipe roughness [m]

    Returns
    -------
    Φ²_LO  [–]  — multiply liquid-only dp by this to get two-phase dp
    """
    x = max(0.0, min(1.0, x))

    Re_lo = G * D / mu_l
    Re_go = G * D / mu_g
    f_lo  = friction_factor(Re_lo, D, roughness)
    f_go  = friction_factor(Re_go, D, roughness)

    # Pressure gradient factors (normalised by G²/D)
    A = f_lo / (2.0 * rho_l)   # liquid-only: (dp/dz)_LO / (G²/D)
    B = f_go / (2.0 * rho_g)   # gas-only:   (dp/dz)_GO / (G²/D)

    # MSH combined gradient
    grad_tp = (A + 2.0 * (B - A) * x) * (1.0 - x) ** (1.0 / 3.0) + B * x**3

    return grad_tp / A   # Φ²_LO


def phi2_lo_friedel(x: float, rho_l: float, rho_g: float,
                    mu_l: float, mu_g: float,
                    sigma: float, G: float, D: float) -> float:
    """Friedel (1979) two-phase friction multiplier."""
    raise NotImplementedError


def phi2_lo_lockhart_martinelli(x: float, rho_l: float, rho_g: float,
                                mu_l: float, mu_g: float) -> float:
    """Lockhart–Martinelli (1949) two-phase friction multiplier."""
    raise NotImplementedError
