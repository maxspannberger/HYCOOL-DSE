# CcH2 (cryo-compressed, supercritical) pipe sizing.
# Operating range: p ∈ [250, 350] bar, T ∈ [25, 110] K — always above p_crit (13 bar).
# Single-phase throughout; structural wall thickness dominates.
# No foam or Al jacket: the 3+ mm SS wall and fluid heat capacity are sufficient.
# Pressure drop is not a sizing driver at 350 bar — 20 bar dp_max used.
# Ref: ASME B31.12; Imperial/arXiv embrittlement data for 316L under gaseous H2.

from dataclasses import dataclass
from math import pi

from hycool_pipe import fluids, friction as fric, materials as mat


@dataclass
class CcH2Config:
    """Design inputs for the CcH2 supercritical pipe."""
    # Flow
    m_dot: float = 0.120       # kg/s
    L: float = 64.0            # m
    n_bends: int = 10

    # Thermodynamic inlet state
    p_in: float = 350e5        # Pa  — 350 bar
    T_in: float = 35.0         # K

    # Design constraints
    dp_max: float = 100.0e5    # Pa  — 100 bar (max: p_exit must stay ≥ 250 bar operating floor)
    T_exit_max: float = 110.0  # K   — upper bound of CcH2 operating envelope

    # Structural
    safety_factor: float = 2.5
    h2_exposure: str = "gaseous"   # gaseous H2 — tighter embrittlement de-rating
    t_ss_min: float = 3e-4         # m  — manufacturing floor (0.3 mm SS)

    # No foam or Al jacket for CcH2
    t_al: float = 0.0

    # Ambient
    T_ambient: float = 315.0   # K  — hot-ground sizing case

    # Material densities
    rho_ss: float = 8000.0

    # Hydraulic
    k_minor: float = 0.02
    roughness: float = 1.5e-6   # m  — SS-316L drawn tubing

    # Fluid
    fluid: str = "hydrogen"

    # t_ss is set by structural sizing in solver.size(); default here is overridden
    t_ss: float = 3e-4


def run_analysis(cfg: CcH2Config | None = None, verbose: bool = True) -> dict:
    """Size the CcH2 pipe: D from dp_max, t_ss from hoop stress.

    No foam or Al jacket — structural SS wall sufficient for thermal margin.

    Returns
    -------
    dict with D, t_ss, geometry, mass, hydraulics, thermal, structural fields.
    """
    import CoolProp.CoolProp as CP
    from hycool_pipe.solver import size
    from dataclasses import replace

    if cfg is None:
        cfg = CcH2Config()

    D, t_ss, t_foam, res = size(cfg, n_seg=200, verbose=verbose)

    # Verify supercritical throughout
    p_crit = CP.PropsSI("pcrit", cfg.fluid)
    assert cfg.p_in > p_crit, "p_in must exceed critical pressure for CcH2 case"

    # Inlet properties
    h0   = CP.PropsSI("Hmass", "P", cfg.p_in, "T", cfg.T_in, cfg.fluid)
    rho0 = CP.PropsSI("D",     "P", cfg.p_in, "Hmass", h0, cfg.fluid)
    mu0  = CP.PropsSI("V",     "P", cfg.p_in, "Hmass", h0, cfg.fluid)
    Re0  = fric.reynolds(D, cfg.m_dot, mu0)
    f0   = fric.friction_factor(Re0, D, cfg.roughness)
    G    = cfg.m_dot / (pi * D**2 / 4)
    u0   = G / rho0

    # Sigma_allow at inlet conditions
    sig      = mat.sigma_allow(cfg.T_in, cfg.h2_exposure)
    sig_hoop = cfg.p_in * D / (2.0 * t_ss)

    # Mass per meter — SS only; no foam or Al jacket
    cfg_m   = replace(cfg, t_ss=t_ss)
    r_i     = D / 2.0
    r_o_ss  = r_i + t_ss
    m_ss    = cfg_m.rho_ss * pi * (r_o_ss**2 - r_i**2)
    m_total = m_ss

    r_o_al = r_o_ss   # outer surface is SS wall (no foam or jacket)

    result = {
        # Geometry
        "D_m":                  D,
        "D_mm":                 D * 1e3,
        "t_ss_m":               t_ss,
        "t_ss_mm":              t_ss * 1e3,
        "OD_mm":                2 * r_o_ss * 1e3,
        # Mass
        "m_ss_kg_per_m":        m_ss,
        "m_total_kg_per_m":     m_total,
        "total_mass_kg":        m_total * cfg.L,
        # Hydraulics
        "Re":                   Re0,
        "f":                    f0,
        "u_inlet_m_s":          u0,
        "dp_total_Pa":          res.dp_total,
        "dp_total_kPa":         res.dp_total / 1e3,
        # Thermal
        "T_exit_K":             res.T_exit,
        "T_exit_max_K":         cfg.T_exit_max,
        "q_total_W":            res.q_total,
        "p_exit_bar":           res.p_exit / 1e5,
        # Structural
        "sigma_allow_MPa":      sig / 1e6,
        "sigma_hoop_MPa":       sig_hoop / 1e6,
        "SF_actual":            sig / sig_hoop,
        "h2_exposure":          cfg.h2_exposure,
    }

    if verbose:
        _print_results(result, cfg)

    return result


def _print_results(res: dict, cfg: CcH2Config) -> None:
    print("\n=== CcH2 Pipe Sizing Results ===")
    print(f"  Fluid: {cfg.fluid}  |  m_dot={cfg.m_dot*1e3:.1f} g/s  |  L={cfg.L:.0f} m")
    print(f"  p_in={cfg.p_in/1e5:.0f} bar  |  T_in={cfg.T_in:.0f} K  |  T_exit_max={cfg.T_exit_max:.0f} K")
    print(f"  dp_max={cfg.dp_max/1e3:.0f} kPa  |  SF={cfg.safety_factor}  |  H2 exposure: {cfg.h2_exposure}")
    print()
    print(f"  Hydraulics:")
    print(f"    D        = {res['D_mm']:.2f} mm")
    print(f"    Re       = {res['Re']:.0f}  |  f = {res['f']:.5f}  |  u = {res['u_inlet_m_s']:.2f} m/s")
    print(f"    dp_total = {res['dp_total_kPa']:.1f} kPa  (limit: {cfg.dp_max/1e3:.0f} kPa)")
    print()
    print(f"  Structural (dominant):")
    print(f"    sigma_allow  = {res['sigma_allow_MPa']:.1f} MPa  ({cfg.h2_exposure} H2 de-rating applied)")
    print(f"    t_ss         = {res['t_ss_mm']:.3f} mm  (hoop stress + SF {cfg.safety_factor})")
    print(f"    sigma_hoop   = {res['sigma_hoop_MPa']:.1f} MPa  (actual at sized wall)")
    print(f"    SF_actual    = {res['SF_actual']:.2f}")
    print()
    print(f"  Thermal (no foam — SS wall + fluid capacity sufficient):")
    print(f"    T_exit   = {res['T_exit_K']:.2f} K  (limit: {res['T_exit_max_K']:.0f} K, margin: {res['T_exit_max_K']-res['T_exit_K']:.1f} K)")
    print(f"    q_total  = {res['q_total_W']:.0f} W")
    print(f"    p_exit   = {res['p_exit_bar']:.2f} bar")
    print()
    print(f"  Geometry (OD = {res['OD_mm']:.1f} mm, bare SS pipe):")
    print(f"    m_ss   = {res['m_ss_kg_per_m']:.4f} kg/m")
    print(f"    m/L    = {res['m_total_kg_per_m']:.4f} kg/m")
    print(f"    total  = {res['total_mass_kg']:.2f} kg  (over {cfg.L:.0f} m)")
    print()
