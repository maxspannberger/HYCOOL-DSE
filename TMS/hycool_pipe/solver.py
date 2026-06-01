# Segmented (p, h) marching scheme and outer (D, t_foam) sizing solver.
# Two-phase: MSH friction multiplier + Chen boiling h-coefficient (point 7).
#
# Orientation: all segments default to horizontal (inclination_deg=0).
# geometry.py will supply per-segment angles once routing is implemented.

from math import pi, sin, radians
from dataclasses import dataclass, field
import warnings

import CoolProp.CoolProp as CP

from hycool_pipe import fluids, friction as fric, network, boiling as boil
from hycool_pipe import materials as mat, ambient as amb
from hycool_pipe.regimes import classify_regime, FlowRegime, two_phase_pattern, FlowPattern
from scipy.optimize import brentq


# ── Exceptions ───────────────────────────────────────────────────────────────

class ChokedFlow(Exception):
    def __init__(self, segment_index: int, mach: float):
        self.segment_index = segment_index
        self.mach = mach
        super().__init__(
            f"Choked flow at segment {segment_index} (M={mach:.3f} ≥ 1)"
        )


class PhaseTransition(Warning):
    """Kept for backwards compatibility; no longer raised by march() after point 7."""
    pass


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class MarchResult:
    """Output of a single march() call."""
    # Per-segment arrays (length = n_seg)
    z:       list = field(default_factory=list)   # m    — axial position (segment start)
    p:       list = field(default_factory=list)   # Pa
    T:       list = field(default_factory=list)   # K
    h:       list = field(default_factory=list)   # J/kg
    rho:     list = field(default_factory=list)   # kg/m³
    u:       list = field(default_factory=list)   # m/s
    Re:      list = field(default_factory=list)
    f:       list = field(default_factory=list)   # Darcy friction factor (or f_LO in 2-ph)
    phi2:    list = field(default_factory=list)   # MSH multiplier (1.0 in single-phase)
    Q_seg:   list = field(default_factory=list)   # W  — heat into fluid per segment
    h_inner: list = field(default_factory=list)   # W/m²K — inner film coefficient
    x:       list = field(default_factory=list)   # vapour quality (-1 / 0–1 / 2)
    regime:  list = field(default_factory=list)   # FlowRegime string
    pattern: list = field(default_factory=list)   # FlowPattern string (two-phase only)
    k_foam_seg: list = field(default_factory=list)  # W/mK — foam conductivity per segment

    # Exit scalars
    p_exit: float = 0.0   # Pa
    T_exit: float = 0.0   # K
    h_exit: float = 0.0   # J/kg
    x_exit: float = -1.0  # quality at exit

    # Totals
    dp_total: float = 0.0   # Pa
    q_total:  float = 0.0   # W  — Σ Q_seg


# ── Inner march ──────────────────────────────────────────────────────────────

def march(
    D: float,
    t_foam: float,
    cfg,                    # LH2Config or CcH2Config
    n_seg: int = 200,
    inclination_deg: float = 0.0,   # positive = uphill; per-segment array support TODO
    plumbing_margin: float = 1.0,   # multiply Q_seg by this factor (1.3–1.5 per NASA)
    p_min: float = 5e3,             # Pa — stop march early if pressure drops below this
) -> MarchResult:
    """Segmented (p, h) march along the pipe.

    State is tracked in (p, h) space throughout — safe inside the two-phase dome.
    Two-phase friction multiplier (Φ²_LO) is not yet applied; added in point 7.

    Parameters
    ----------
    D             : inner bore diameter [m]
    t_foam        : foam insulation thickness [m]
    cfg           : LH2Config (or CcH2Config) — supplies fluid, geometry, constraints
    n_seg         : number of axial segments (default 200)
    inclination_deg : pipe angle from horizontal [°]; positive = climbing
                      scalar for now; geometry.py will supply per-segment array later
    plumbing_margin : global heat-leak multiplier (NASA 1.3–1.5×)

    Returns
    -------
    MarchResult
    """
    fluid    = cfg.fluid
    m_dot    = cfg.m_dot
    roughness = cfg.roughness
    T_amb_val = cfg.T_ambient
    dz       = cfg.L / n_seg
    theta    = radians(inclination_deg)
    A        = pi * D**2 / 4.0
    g        = 9.81  # m/s²

    # Outer jacket radius — needed for h_outer (natural convection)
    r_o_al = D / 2.0 + cfg.t_ss + t_foam + cfg.t_al
    D_outer = 2.0 * r_o_al

    # Initial state from (p_in, T_in)
    h_in = CP.PropsSI("Hmass", "P", cfg.p_in, "T", cfg.T_in, fluid)
    p = cfg.p_in
    h = h_in

    res = MarchResult()
    q_total = 0.0

    for i in range(n_seg):
        # ── Properties at segment start ───────────────────────────────────────
        props = fluids.get_props(fluid, p, h)
        x_val = props["x"]
        T     = props["T"]
        G     = m_dot / A    # mass flux [kg/m²s]

        regime = classify_regime(x_val)

        # ── Regime-specific hydraulics & heat transfer ────────────────────────
        if regime == FlowRegime.SATURATED_TWO_PHASE:
            # Saturation properties (avoids CoolProp mixture ambiguity)
            rho_l = fluids.rho_sat_liq(fluid, p)
            rho_g = fluids.rho_sat_vap(fluid, p)
            mu_l  = fluids.mu_sat_liq(fluid, p)
            mu_g  = fluids.mu_sat_vap(fluid, p)

            # HEM mixture density and velocity for reporting
            rho = 1.0 / (x_val / rho_g + (1.0 - x_val) / rho_l)
            u   = G / rho

            # Liquid-only friction (MSH baseline)
            Re_val = G * D / mu_l
            f_val  = fric.friction_factor(Re_val, D, roughness)

            # MSH two-phase multiplier
            phi2_val = fric.phi2_lo_msh(x_val, G, D, rho_l, rho_g, mu_l, mu_g, roughness)

            # Two-phase pressure drop: Φ²_LO × (dp/dz)_LO
            dp_lo       = f_val * (dz / D) * G**2 / (2.0 * rho_l)
            dp_friction = phi2_val * dp_lo

            # Chen inner film coefficient
            h_in = boil.h_chen(G, x_val, D, fluid, p)

        else:
            # Single-phase (subcooled liquid or superheated vapour)
            rho   = props["rho"]
            mu    = props["mu"]
            u     = G / rho
            phi2_val = 1.0
            rho_l = rho   # dummy — only used in two-phase pattern call
            rho_g = rho

            Re_val = fric.reynolds(D, m_dot, mu)
            f_val  = fric.friction_factor(Re_val, D, roughness)
            dp_friction = f_val * (dz / D) * 0.5 * rho * u**2

            # Dittus-Boelter inner film coefficient
            Pr   = props["cp"] * mu / props["k"]
            h_in = boil.h_dittus_boelter(Re_val, Pr, props["k"], D)

            # Choking check (single-phase; two-phase sound speed in point 8)
            a    = fluids.sound_speed(fluid, p, h)
            mach = u / a
            if mach >= 1.0:
                raise ChokedFlow(i, mach)

        # ── Heat ingress ──────────────────────────────────────────────────────
        # k_foam evaluated at log-mean temperature of the foam shell.
        # h_outer from Churchill–Chu natural convection (altitude=0 for hot-ground).
        T_bar_foam = mat.T_log_mean(T, T_amb_val)
        k_foam_seg_val = mat.k_foam(T_bar_foam, getattr(cfg, 'foam_aging_factor', 1.3))
        k_ss_seg = mat.k_SS(T)
        k_al_seg = mat.k_Al(T_amb_val)  # Al jacket near ambient temperature
        h_out = amb.h_outer(altitude_m=0.0, D_outer=D_outer,
                            T_surface=T_amb_val - 5.0)  # outer surface ≈ T_amb − small drop

        R_pm  = network.resistance_per_length(
            D, cfg.t_ss, t_foam, cfg.t_al,
            k_ss=k_ss_seg,
            k_foam_val=k_foam_seg_val,
            k_al=k_al_seg,
            h_inner_coef=h_in,
            h_outer_coef=h_out,
        )
        Q_seg = (T_amb_val - T) / R_pm * dz * plumbing_margin

        # ── Gravity pressure drop ──────────────────────────────────────────────
        dp_gravity = rho * g * dz * sin(theta)   # 0 for horizontal
        dp_seg     = dp_friction + dp_gravity

        # ── Flow-pattern map (two-phase only) ────────────────────────────────
        if regime == FlowRegime.SATURATED_TWO_PHASE:
            pattern_val = two_phase_pattern(G, x_val, rho_l, rho_g, D)
        else:
            pattern_val = FlowPattern.SINGLE_PHASE

        # ── Store segment ─────────────────────────────────────────────────────
        res.z.append(i * dz)
        res.p.append(p)
        res.T.append(T)
        res.h.append(h)
        res.rho.append(rho)
        res.u.append(u)
        res.Re.append(Re_val)
        res.f.append(f_val)
        res.phi2.append(phi2_val)
        res.Q_seg.append(Q_seg)
        res.h_inner.append(h_in)
        res.x.append(x_val)
        res.regime.append(regime)
        res.pattern.append(pattern_val)
        res.k_foam_seg.append(k_foam_seg_val)
        q_total += Q_seg

        # ── Advance state ─────────────────────────────────────────────────────
        h = h + Q_seg / m_dot
        p = p - dp_seg

        # Stop early if pressure is unphysically low (undersized pipe)
        if p < p_min:
            res.p_exit   = p_min
            res.h_exit   = h
            res.T_exit   = float("nan")
            res.x_exit   = float("nan")
            res.dp_total = cfg.p_in - p_min
            res.q_total  = q_total
            return res

    # ── Exit state ────────────────────────────────────────────────────────────
    res.p_exit   = p
    res.h_exit   = h
    res.T_exit   = CP.PropsSI("T",      "P", p, "Hmass", h, fluid)
    res.x_exit   = fluids.quality(fluid, p, h)
    res.dp_total = cfg.p_in - p
    res.q_total  = q_total

    return res


# ── Structural wall sizing ───────────────────────────────────────────────────

def size_t_ss(
    D: float,
    p_design: float,
    SF: float,
    T: float,
    h2_exposure: str = "liquid",
    t_min: float = 3e-4,
) -> float:
    """Minimum SS-316L inner wall thickness [m] from hoop stress.

    σ_h = p·D / (2·t)  ≤  σ_allow / SF
    → t = p·D·SF / (2·σ_allow)  , clamped to manufacturing floor t_min.

    Parameters
    ----------
    D           : inner bore diameter [m]
    p_design    : design pressure [Pa]
    SF          : safety factor (≥ 2.5 for crewed aircraft)
    T           : fluid temperature for σ_allow lookup [K]
    h2_exposure : "liquid" (LH2) or "gaseous" (CcH2) — affects de-rating
    t_min       : manufacturing floor [m]  (0.3 mm SS default)
    """
    from hycool_pipe.materials import sigma_allow
    t_structural = p_design * D * SF / (2.0 * sigma_allow(T, h2_exposure))
    return max(t_structural, t_min)


# ── Outer sizing solver ──────────────────────────────────────────────────────

def size(cfg, n_seg: int = 200, plumbing_margin: float = 1.0,
         verbose: bool = True) -> tuple:
    """Find (D, t_ss, t_foam) satisfying the case constraints.

    Step 1 — D from dp_max (both cases).
    Step 2 — t_ss from hoop stress at the sized D (CcH2: several mm; LH2: manufacturing floor).
    Step 3 — t_foam from thermal constraint:
              LH2:  q_total  = cfg.q_total_max_W
              CcH2: T_exit   = cfg.T_exit_max

    Returns
    -------
    (D, t_ss, t_foam, result)
    """
    is_cch2 = hasattr(cfg, "T_exit_max")
    # ── Step 1: size D from pressure-drop constraint ──────────────────────────
    if verbose:
        print("Sizing D from dp constraint ...", end=" ", flush=True)

    _t_foam_dummy = 0.0 if is_cch2 else 0.05   # CcH2 runs bare SS; LH2 needs a finite foam guess

    def dp_residual(D):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PhaseTransition)
            try:
                r = march(D, t_foam=_t_foam_dummy, cfg=cfg, n_seg=n_seg,
                          plumbing_margin=plumbing_margin)
            except ChokedFlow:
                return 1e9  # D too small → huge positive residual
        return r.dp_total - cfg.dp_max

    # Use analytical Darcy–Weisbach to bracket D without risk of march going negative.
    import CoolProp.CoolProp as _CP
    _h0   = _CP.PropsSI("Hmass", "P", cfg.p_in, "T", cfg.T_in, cfg.fluid)
    _rho0 = _CP.PropsSI("D",     "P", cfg.p_in, "Hmass", _h0,  cfg.fluid)
    _mu0  = _CP.PropsSI("V",     "P", cfg.p_in, "Hmass", _h0,  cfg.fluid)

    def _dp_analytical(D):
        Re  = fric.reynolds(D, cfg.m_dot, _mu0)
        f   = fric.friction_factor(Re, D, cfg.roughness)
        k_t = cfg.k_minor * cfg.n_bends
        return (8.0 / (pi**2 * _rho0)) * (cfg.m_dot**2 / D**4) * (f * cfg.L / D + k_t)

    lo_D, hi_D = 2e-3, 0.20
    dp_lo = _dp_analytical(lo_D)
    dp_hi = _dp_analytical(hi_D)

    if dp_lo < cfg.dp_max:
        raise ValueError(
            f"dp at D={lo_D*1e3:.0f} mm ({dp_lo/1e3:.1f} kPa) is already below dp_max "
            f"({cfg.dp_max/1e3:.0f} kPa). Reduce dp_max or increase m_dot."
        )
    if dp_hi > cfg.dp_max:
        raise ValueError(
            f"dp at D={hi_D*1e3:.0f} mm ({dp_hi/1e3:.1f} kPa) exceeds dp_max even at "
            f"D={hi_D*1e3:.0f} mm. Increase dp_max or decrease m_dot."
        )

    D = brentq(dp_residual, lo_D, hi_D, xtol=1e-6)
    if verbose:
        print(f"D = {D*1e3:.2f} mm")

    # ── Step 2: structural wall thickness ─────────────────────────────────────
    SF          = getattr(cfg, "safety_factor", 2.5)
    h2_exp      = getattr(cfg, "h2_exposure",   "liquid")
    t_ss_floor  = getattr(cfg, "t_ss_min",       3e-4)
    t_ss = size_t_ss(D, cfg.p_in, SF, cfg.T_in, h2_exp, t_ss_floor)
    if verbose:
        governed = "hoop stress" if t_ss > t_ss_floor * 1.01 else "manuf. floor"
        print(f"t_ss = {t_ss*1e3:.3f} mm  ({governed})")

    # Build a config copy with the structurally-sized t_ss for foam sizing
    from dataclasses import replace as dc_replace
    cfg_struct = dc_replace(cfg, t_ss=t_ss)

    # ── Step 3: foam from thermal constraint (LH2 only) ─────────────────────────
    if is_cch2:
        # No foam or Al jacket for CcH2 — structural SS wall and fluid heat capacity sufficient.
        # T_exit constraint has ~57 K of margin even with bare SS pipe.
        t_foam = 0.0
        if verbose:
            print("t_foam = 0 mm  (no insulation — CcH2 thermal constraint non-binding)")
        result = march(D, t_foam, cfg=cfg_struct, n_seg=n_seg,
                       plumbing_margin=plumbing_margin)
        return D, t_ss, t_foam, result

    if verbose:
        print(f"Sizing t_foam from q_total constraint ...", end=" ", flush=True)

    def thermal_residual(t_foam):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PhaseTransition)
            r = march(D, t_foam, cfg=cfg_struct, n_seg=n_seg,
                      plumbing_margin=plumbing_margin)
        return r.q_total - cfg.q_total_max_W

    lo_t, hi_t = 1e-4, 0.50
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PhaseTransition)
        res_hi = march(D, hi_t, cfg_struct, n_seg, plumbing_margin=plumbing_margin)

    if res_hi.q_total > cfg.q_total_max_W:
        raise ValueError(f"q_total at t_foam={hi_t*1e3:.0f} mm still exceeds limit.")

    t_foam = brentq(thermal_residual, lo_t, hi_t, xtol=1e-6)

    if verbose:
        print(f"t_foam = {t_foam*1e3:.2f} mm")

    # ── Final march at fully sized geometry ───────────────────────────────────
    result = march(D, t_foam, cfg=cfg_struct, n_seg=n_seg,
                   plumbing_margin=plumbing_margin)
    return D, t_ss, t_foam, result
