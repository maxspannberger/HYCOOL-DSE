# Thermal resistance network, parasitic support conduction, plumbing margin.
# Ref: NASA "+ Additional Fuel System Component = 1.5(Tank + Insulation)"

from math import pi, log

# Approximate fixed material conductivities used until materials.py is wired in (point 8).
# These are mid-range cryogenic values; temperature-dependent replacements come later.
_K_SS_APPROX = 14.0    # W/mK — SS-316L (conservative mid-cryogenic value)
_K_AL_APPROX = 130.0   # W/mK — Al-6061 (mid-cryogenic; drops to ~30 at 20 K)


class VacuumJacketInsulation:
    # Out of scope — architecture hook for future implementation.
    def resistance(self, *args, **kwargs):
        raise NotImplementedError("Vacuum jacket variant is out of scope (§9).")


class MLIInsulation:
    # Out of scope — architecture hook for future implementation.
    def resistance(self, *args, **kwargs):
        raise NotImplementedError("MLI variant is out of scope (§9).")


def resistance_per_length(
    D_inner: float,
    t_ss: float,
    t_foam: float,
    t_al: float,
    k_ss: float = _K_SS_APPROX,
    k_foam_val: float = 0.0225,
    k_al: float = _K_AL_APPROX,
    h_inner_coef: float = 0.0,
    h_outer_coef: float = 0.0,
) -> float:
    """Total thermal resistance per unit length [K·m/W] for the pipe cross-section.

    Layers (inside → outside):
        fluid film   1 / (h_inner * 2π r_i)
        SS wall      ln(r_o_ss / r_i_ss) / (2π k_SS)
        PU foam      ln(r_o_foam / r_i_foam) / (2π k_foam)
        Al jacket    ln(r_o_al / r_i_al) / (2π k_Al)
        outer film   1 / (h_outer * 2π r_o_al)

    Convective film coefficients h_inner and h_outer default to 0 (infinite resistance
    terms skipped) — they are added in point 8 when boiling.py and ambient.py are wired.

    Parameters
    ----------
    D_inner   : inner bore diameter [m]
    t_ss      : SS wall thickness [m]
    t_foam    : foam insulation thickness [m]
    t_al      : Al jacket thickness [m]
    k_ss      : SS thermal conductivity [W/mK]
    k_foam_val: effective foam conductivity (base × aging factor) [W/mK]
    k_al      : Al thermal conductivity [W/mK]
    h_inner_coef : inner film coefficient [W/m²K]  (0 → skip)
    h_outer_coef : outer film coefficient [W/m²K]  (0 → skip)

    Returns
    -------
    R  [K·m/W]  — multiply by dz to get segment resistance [K/W]
    """
    r_i  = D_inner / 2.0
    r_ss = r_i    + t_ss
    r_fo = r_ss   + t_foam
    r_al = r_fo   + t_al

    R = 0.0

    # Inner convective film (skipped when h_inner_coef == 0)
    if h_inner_coef > 0.0:
        R += 1.0 / (h_inner_coef * 2.0 * pi * r_i)

    # SS wall conduction
    R += log(r_ss / r_i) / (2.0 * pi * k_ss)

    # Foam insulation (dominant term)
    R += log(r_fo / r_ss) / (2.0 * pi * k_foam_val)

    # Al jacket conduction
    R += log(r_al / r_fo) / (2.0 * pi * k_al)

    # Outer convective film (skipped when h_outer_coef == 0)
    if h_outer_coef > 0.0:
        R += 1.0 / (h_outer_coef * 2.0 * pi * r_al)

    return R


def q_parasitic(n_supports: int, kA_over_L_support: float, delta_T: float) -> float:
    """Lumped parasitic conduction [W] through pipe supports / fittings.

    Q_parasitic = N_supports × (kA/L)_support × ΔT
    """
    return n_supports * kA_over_L_support * delta_T
