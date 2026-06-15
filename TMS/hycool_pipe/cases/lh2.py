# LH2 (sub-critical) pipe sizing — single-phase baseline
# Properties from CoolProp; friction from Colebrook–White.
# Segmented two-phase marching is added in point 6+.

from dataclasses import dataclass, field
from math import pi, log, exp
from scipy.optimize import brentq

from hycool_pipe import fluids, friction as fric


@dataclass
class LH2Config:
    """Design inputs for the LH2 sub-critical pipe."""
    # Flow
    m_dot: float = 0.120       # kg/s — mass flow per line
    L: float = 64.0            # m    — straight-line length (routing factor applied separately)
    n_bends: int = 10

    # Thermodynamic inlet state
    p_in: float = 2.0e5        # Pa   — inlet pressure (~2 bar tank head)
    T_in: float = 20.0         # K    — inlet temperature

    # Design constraints
    dp_max: float = 0.5e5      # Pa  — 0.5 bar pressure-drop limit
    q_total_max_W: float = 2000.0  # W  — max total heat ingress over full pipe length

    # Ambient (hot-ground sizing case per §1 issue 6)
    T_ambient: float = 315.0   # K

    # Insulation
    k_foam_base: float = 0.0173   # W/mK — base PUF conductivity
    foam_aging_factor: float = 1.3  # moisture / cryopumping aging
    rho_foam: float = 40.0         # kg/m³

    # Structural wall thicknesses — manufacturing floors only (hoop stress ≈ 0 at 2 bar)
    # SS-316L drawn tubing: 0.3 mm is standard aerospace minimum for weld quality / handling
    # Al jacket: foam containment + vapour barrier; 0.3 mm thin-wall drawn tube is standard
    t_ss: float = 3.0e-4       # m — SS-316L inner pipe wall (0.3 mm manufacturing floor)
    t_al: float = 3.0e-4       # m — Al-6061 outer jacket (0.3 mm manufacturing floor)
    rho_ss: float = 8000.0     # kg/m³
    rho_al: float = 2700.0     # kg/m³

    # Hydraulic
    k_minor: float = 0.02      # minor-loss coefficient per bend
    roughness: float = 1.5e-6  # m — SS-316L drawn tubing

    # Fluid
    fluid: str = fluids.DEFAULT_FLUID   # "hydrogen"; switch to "parahydrogen" if needed

    @property
    def k_foam(self) -> float:
        return self.k_foam_base * self.foam_aging_factor


# ── Fluid properties at inlet ────────────────────────────────────────────────

def _inlet_props(cfg: LH2Config) -> dict:
    """CoolProp properties at the pipe inlet (p_in, T_in)."""
    import CoolProp.CoolProp as CP
    h_in = CP.PropsSI("Hmass", "P", cfg.p_in, "T", cfg.T_in, cfg.fluid)
    return fluids.get_props(cfg.fluid, cfg.p_in, h_in)


# ── Pressure drop ────────────────────────────────────────────────────────────

def _dp(D: float, rho: float, mu: float, cfg: LH2Config) -> float:
    """Darcy–Weisbach pressure drop [Pa] with Colebrook friction factor."""
    Re  = fric.reynolds(D, cfg.m_dot, mu)
    f   = fric.friction_factor(Re, D, cfg.roughness)
    k_t = cfg.k_minor * cfg.n_bends
    return (8.0 / (pi**2 * rho)) * (cfg.m_dot**2 / D**4) * (f * cfg.L / D + k_t)


def _solve_diameter(rho: float, mu: float, cfg: LH2Config) -> float:
    """Bisect D ∈ [2 mm, 200 mm] so that dp(D) = dp_max.

    At each candidate D the friction factor is recomputed from Re(D),
    so the Colebrook iteration is nested inside the bisection.
    """
    def residual(D):
        return _dp(D, rho, mu, cfg) - cfg.dp_max

    lo, hi = 2e-3, 0.20
    if residual(lo) * residual(hi) > 0:
        dp_lo = _dp(lo, rho, mu, cfg)
        dp_hi = _dp(hi, rho, mu, cfg)
        raise ValueError(
            f"No solution for D in [{lo*1e3:.0f}, {hi*1e3:.0f}] mm. "
            f"dp(lo)={dp_lo/1e3:.1f} kPa, dp(hi)={dp_hi/1e3:.1f} kPa. "
            f"Check dp_max and m_dot."
        )
    return brentq(residual, lo, hi, xtol=1e-7)


# ── Insulation thickness ─────────────────────────────────────────────────────

def _solve_insulation(D: float, T_fluid: float, cfg: LH2Config) -> float:
    """Analytical t_foam [m] so that total conductive heat ingress = q_total_max_W.

    Cylindrical conduction only (convective film resistances added in point 8).
    """
    r1 = D / 2.0 + cfg.t_ss
    exponent = (cfg.T_ambient - T_fluid) * 2.0 * pi * cfg.k_foam * cfg.L / cfg.q_total_max_W
    return r1 * (exp(exponent) - 1.0)


# ── Mass per unit length ─────────────────────────────────────────────────────

def _mass_per_meter(D: float, t_foam: float,
                    cfg: LH2Config) -> tuple[float, float, float, float]:
    """Returns (m_ss, m_foam, m_al, m_total) in kg/m."""
    r_i_ss   = D / 2.0
    r_o_ss   = r_i_ss   + cfg.t_ss
    r_o_foam = r_o_ss   + t_foam
    r_o_al   = r_o_foam + cfg.t_al

    m_ss   = cfg.rho_ss   * pi * (r_o_ss**2   - r_i_ss**2)
    m_foam = cfg.rho_foam * pi * (r_o_foam**2  - r_o_ss**2)
    m_al   = cfg.rho_al   * pi * (r_o_al**2   - r_o_foam**2)
    return m_ss, m_foam, m_al, m_ss + m_foam + m_al


# ── Main entry point ─────────────────────────────────────────────────────────

def run_analysis(cfg: LH2Config | None = None, verbose: bool = True) -> dict:
    """Size the LH2 pipe for the given design config.

    Solves for D (from dp_max constraint) and t_foam (from boil-off constraint)
    using CoolProp fluid properties and Colebrook friction factor.

    Returns
    -------
    dict with keys:
        D_m, D_mm, t_foam_m, t_foam_mm,
        t_ss_m, t_al_m, r_o_al_m (outer radius of jacket),
        m_ss_kg_per_m, m_foam_kg_per_m, m_al_kg_per_m, m_total_kg_per_m,
        total_mass_kg,
        dp_Pa, dp_kPa,
        Re, f,
        q_dot_W_per_m, q_total_W,
        boil_off_frac_per_m, boil_off_frac_total,
        rho_inlet, T_inlet, h_lat_Pa
    """
    if cfg is None:
        cfg = LH2Config()

    props   = _inlet_props(cfg)
    rho     = props["rho"]
    mu      = props["mu"]
    T_fluid = props["T"]
    h_lat_v = fluids.h_lat(cfg.fluid, cfg.p_in)

    D      = _solve_diameter(rho, mu, cfg)
    t_foam = _solve_insulation(D, T_fluid, cfg)

    m_ss, m_foam, m_al, m_total = _mass_per_meter(D, t_foam, cfg)

    Re_val = fric.reynolds(D, cfg.m_dot, mu)
    f_val  = fric.friction_factor(Re_val, D, cfg.roughness)
    dp     = _dp(D, rho, mu, cfg)

    r1    = D / 2.0 + cfg.t_ss
    R_pm  = log((r1 + t_foam) / r1) / (2.0 * pi * cfg.k_foam)
    q_dot = (cfg.T_ambient - T_fluid) / R_pm
    q_tot = q_dot * cfg.L

    r_o_al = D / 2.0 + cfg.t_ss + t_foam + cfg.t_al

    result = {
        "D_m":                  D,
        "D_mm":                 D * 1e3,
        "t_foam_m":             t_foam,
        "t_foam_mm":            t_foam * 1e3,
        "t_ss_m":               cfg.t_ss,
        "t_al_m":               cfg.t_al,
        "r_o_al_m":             r_o_al,
        "OD_mm":                r_o_al * 2e3,
        "m_ss_kg_per_m":        m_ss,
        "m_foam_kg_per_m":      m_foam,
        "m_al_kg_per_m":        m_al,
        "m_total_kg_per_m":     m_total,
        "total_mass_kg":        m_total * cfg.L,
        "dp_Pa":                dp,
        "dp_kPa":               dp / 1e3,
        "Re":                   Re_val,
        "f":                    f_val,
        "q_dot_W_per_m":        q_dot,
        "q_total_W":            q_tot,
        "q_total_max_W":        cfg.q_total_max_W,
        "rho_inlet":            rho,
        "T_inlet":              T_fluid,
        "h_lat_J_per_kg":       h_lat_v,
    }

    if verbose:
        _print_results(result, cfg)

    return result


def _print_results(res: dict, cfg: LH2Config) -> None:
    print("\n=== LH2 Pipe Sizing Results ===")
    print(f"  Fluid: {cfg.fluid}  |  m_dot={cfg.m_dot*1e3:.1f} g/s  |  L={cfg.L:.0f} m  |  p_in={cfg.p_in/1e5:.1f} bar")
    print(f"  T_ambient={cfg.T_ambient:.0f} K  |  dp_max={cfg.dp_max/1e3:.0f} kPa  |  q_total_max={cfg.q_total_max_W:.0f} W")
    print()
    print(f"  Inlet properties (CoolProp):")
    print(f"    rho     = {res['rho_inlet']:.3f} kg/m³")
    print(f"    h_lat   = {res['h_lat_J_per_kg']/1e3:.1f} kJ/kg")
    print(f"    T_inlet = {res['T_inlet']:.3f} K")
    print()
    print(f"  Hydraulics:")
    print(f"    D       = {res['D_mm']:.2f} mm")
    print(f"    Re      = {res['Re']:.0f}")
    print(f"    f       = {res['f']:.5f}  (Colebrook)")
    print(f"    dp      = {res['dp_kPa']:.2f} kPa  (limit: {cfg.dp_max/1e3:.0f} kPa)")
    print()
    print(f"  Thermal:")
    print(f"    k_foam  = {cfg.k_foam:.4f} W/mK  (base {cfg.k_foam_base} x aging {cfg.foam_aging_factor})")
    print(f"    t_foam  = {res['t_foam_mm']:.1f} mm")
    print(f"    q_dot   = {res['q_dot_W_per_m']:.3f} W/m")
    print(f"    q_total = {res['q_total_W']:.1f} W  (limit: {cfg.q_total_max_W:.0f} W)")
    print()
    print(f"  Geometry (OD = {res['OD_mm']:.1f} mm, incl. foam + Al jacket):")
    print(f"    m_ss    = {res['m_ss_kg_per_m']:.4f} kg/m")
    print(f"    m_foam  = {res['m_foam_kg_per_m']:.4f} kg/m")
    print(f"    m_al    = {res['m_al_kg_per_m']:.4f} kg/m")
    print(f"    m/L     = {res['m_total_kg_per_m']:.4f} kg/m")
    print(f"    total   = {res['total_mass_kg']:.2f} kg  (over {cfg.L:.0f} m)")
    print()
