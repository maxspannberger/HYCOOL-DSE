# Temperature-dependent material properties and structural allowables.
# Refs: NIST Monograph 177 (cryogenic properties);
#       Tseng et al. "Thermal conductivity of polyurethane foams from room temperature to 20 K";
#       ASME B31.12 for sigma_allow.

import numpy as np


# ── Tabulated data ────────────────────────────────────────────────────────────
# SS-316L thermal conductivity [W/mK] vs T [K]  (NIST/ASME cryogenic data)
_SS_T = np.array([4,    10,   20,   40,   60,   80,   100,  150,  200,  250,  300])
_SS_K = np.array([5.9,  6.5,  7.8,  9.2,  10.1, 11.0, 11.8, 12.8, 13.4, 14.0, 14.6])

# Al-6061 thermal conductivity [W/mK] vs T [K]  (NIST/ASME cryogenic data)
_AL_T = np.array([4,    10,   20,   40,   77,   100,  150,  200,  250,  300])
_AL_K = np.array([7.0,  22.0, 43.0, 78.0, 115.0,142.0,164.0,170.0,172.0,167.0])

# Polyurethane foam (PUF) base conductivity [W/mK] vs T [K]  (Tseng et al.)
# Log-mean temperature of the foam shell is used for evaluation (see T_log_mean()).
_FOAM_T = np.array([20,   50,   80,   120,  160,  200,  250,  295])
_FOAM_K = np.array([0.009,0.011,0.013,0.016,0.019,0.021,0.023,0.025])


def _interp(T: float, T_table: np.ndarray, K_table: np.ndarray) -> float:
    """Linear interpolation; clips to table bounds rather than extrapolating."""
    return float(np.interp(T, T_table, K_table))


# ── Conductivity functions ────────────────────────────────────────────────────

def k_SS(T: float) -> float:
    """SS-316L thermal conductivity [W/mK] at temperature T [K]."""
    return _interp(T, _SS_T, _SS_K)


def k_Al(T: float) -> float:
    """Al-6061 thermal conductivity [W/mK] at temperature T [K]."""
    return _interp(T, _AL_T, _AL_K)


def k_foam_base(T: float) -> float:
    """PUF base conductivity [W/mK] at temperature T [K] (Tseng et al., no aging)."""
    return _interp(T, _FOAM_T, _FOAM_K)


def k_foam(T_bar: float, aging_factor: float = 1.3) -> float:
    """Effective PUF conductivity [W/mK] at log-mean foam temperature T_bar [K].

    aging_factor = 1.3 accounts for moisture ingress and cryopumping over service life.
    Use T_log_mean(T_cold, T_hot) to compute T_bar for a cylindrical foam shell.
    """
    return k_foam_base(T_bar) * aging_factor


def T_log_mean(T_cold: float, T_hot: float) -> float:
    """Log-mean temperature [K] of a cylindrical shell with boundary temperatures T_cold, T_hot.

    Representative temperature for evaluating k_foam across the foam thickness.
    T_lm = (T_hot - T_cold) / ln(T_hot / T_cold)
    """
    if T_hot <= T_cold:
        return (T_hot + T_cold) / 2.0
    return (T_hot - T_cold) / np.log(T_hot / T_cold)


# ── Structural allowable ─────────────────────────────────────────────────────

def sigma_allow(T: float, h2_exposure: str = "liquid") -> float:
    """Design allowable stress [Pa] for SS-316L at temperature T [K].

    h2_exposure:
        "liquid"  — LH2 case; ductility loss ~20–30 % under hydrogen exposure.
        "gaseous" — CcH2 case; embrittlement more aggressive; tighter de-rating (~40 %).

    Ref: ASME B31.12 Table A-1; Imperial/arXiv data on 316L at 20 K under H2 exposure.
    Base allowable at 300 K: ~170 MPa (ASME).
    Cryogenic enhancement: yield strength increases ~30 % below 100 K.
    Hydrogen de-rating applied on top.
    """
    # Base ASME B31.12 allowable for 316L annealed [MPa] vs T [K]
    _T_tab   = np.array([20,   77,   100,  150,  200,  250,  300])
    _sig_tab = np.array([220,  215,  210,  195,  185,  175,  170])  # MPa
    base_MPa = _interp(T, _T_tab, _sig_tab)

    if h2_exposure == "gaseous":
        de_rate = 0.60   # 40 % knock-down for gaseous H2 embrittlement
    else:
        de_rate = 0.75   # 25 % knock-down for liquid H2 (less aggressive)

    return base_MPa * de_rate * 1e6  # Pa
