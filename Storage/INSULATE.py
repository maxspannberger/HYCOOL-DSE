"""
INSULATE.py
========================
MLI vacuum-jacket sizing for a liquid-hydrogen (LH2) aircraft tank, following
the heat-leak budget approach of:

    P. Virdi, W. Guo, L. Cattafesta III, et al.,
    "Design and optimization of a liquid-hydrogen storage tank and thermal
     management system", Applied Energy 393 (2025) 126054.

--------------------------------------------------------------------------
THE METHOD
--------------------------------------------------------------------------
1. Maximum allowable heat-leak rate (pressure-rise constraint):

       Q_leak = (E_f - E_i) / tau_H

   E_i = internal energy of the LH2 just after filling (state at P0)
   E_f = internal energy when the tank pressure reaches P_vent
   The tank is rigid and sealed, so V and total H2 mass m are constant and
   E_f - E_i is simply the heat absorbed along the constant-volume isochore
   from P0 to P_vent:   E_f - E_i = m * (u(P_vent, rho) - u(P0, rho)),
   with rho = m/V fixed.

2. MLI blanket sizing (Lockheed equation):
   Solve for the number of reflector layers N that achieves the target
   heat flux q_target = Q_leak / A.

3. Outer (vacuum-jacket) surface temperature from natural convection of
   ambient air:

       Q_leak = h_air * A * (T0 - T_s)          (T0 = 293 K ambient)

   h_air from the standard Nusselt correlation for natural convection on a
   horizontal cylinder (Churchill-Chu).

--------------------------------------------------------------------------
WHAT THE PAPER LEAVES TO REFERENCES (clearly flagged below)
--------------------------------------------------------------------------
  * hydrogen properties  -> CoolProp (preferred) or a documented approximate
                            para-hydrogen saturation table fallback.
  * h_air correlation    -> Ref [35]; Churchill-Chu horizontal-cylinder.

Only numpy is required. CoolProp is used automatically if installed.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

# Optional, strongly recommended for accurate H2 thermodynamics.
try:
    from CoolProp.CoolProp import PropsSI
    _HAS_COOLPROP = True
except Exception:                                   # pragma: no cover
    _HAS_COOLPROP = False


# =====================================================================
# 1. TANK GEOMETRY
#    Cylindrical shell (radius a; length l_s) + two hemispherical end
#    caps of radius a.
#    Shape ratios:  phi = a/c ,  lambda = l_s / (l_s + 2 c).
#    Volume constraint:  V = (4/3) pi a c^2 + pi a c l_s.
# =====================================================================
@dataclass
class TankGeometry:
    a: float        # cylinder / hemisphere-cap radius [m]
    c: float        # minor radius of the cross-section / cap height [m]
    l_s: float      # length of the cylindrical shell section [m]
    V: float        # enclosed volume [m^3]
    A_out: float    # outer surface area [m^2]


def build_tank(V: float, phi: float, lam: float) -> TankGeometry:
    """
    Solve the tank dimensions from the required volume V and the two
    dimensionless shape ratios phi = a/c and lambda = l_s/(l_s + 2c),
    then return the geometry including the outer surface area A_out of a
    cylinder with hemispherical end caps (radius a, length l_s).
    """
    if not (0.0 < lam < 1.0):
        raise ValueError("lambda must be in (0, 1)")
    if phi <= 0:
        raise ValueError("phi must be positive")

    # a = phi*c ; l_s = 2*c*lam/(1-lam)
    # V = pi*phi*c^3 * [ 4/3 + 2*lam/(1-lam) ]  ->  solve for c.
    shape_factor = (4.0 / 3.0) + 2.0 * lam / (1.0 - lam)
    c = (V / (math.pi * phi * shape_factor)) ** (1.0 / 3.0)
    a = phi * c
    l_s = 2.0 * c * lam / (1.0 - lam)

    # Outer area = cylinder lateral area + two hemispherical caps (= one sphere).
    A_out = 2.0 * math.pi * a * l_s + 4.0 * math.pi * a * a
    return TankGeometry(a=a, c=c, l_s=l_s, V=V, A_out=A_out)


# =====================================================================
# 2. HYDROGEN THERMODYNAMICS  ->  pressure-rise heat-leak constraint
#    Q_leak = m * (u(P_vent, rho) - u(P0, rho)) / tau_H,  rho = m/V fixed.
# =====================================================================
FLUID = "ParaHydrogen"          # NASA/paper use para-H2 for LH2 storage


def _sat_props_coolprop(P: float):
    """Saturated-liquid/-vapor density and internal energy at pressure P [Pa]."""
    rho_l = PropsSI("D", "P", P, "Q", 0, FLUID)
    rho_g = PropsSI("D", "P", P, "Q", 1, FLUID)
    u_l = PropsSI("U", "P", P, "Q", 0, FLUID)
    u_g = PropsSI("U", "P", P, "Q", 1, FLUID)
    return rho_l, rho_g, u_l, u_g


# --- Fallback table (used only when CoolProp is unavailable) ----------
# Approximate saturated para-hydrogen properties (NIST-class values).
# Internal energy referenced to u_l = 0 at the 1.013 bar saturation point.
# REPLACE with CoolProp for design-grade accuracy.
_PH2_TABLE = {
    # P[Pa]:   (T_sat[K], rho_l[kg/m3], rho_g[kg/m3], u_l[kJ/kg], u_g[kJ/kg])
    101300.0: (20.28, 70.85, 1.331, 0.0,  371.9),
    150000.0: (22.10, 68.70, 1.910, 18.1, 360.0),
    200000.0: (23.53, 66.90, 2.490, 32.0, 348.0),
    300000.0: (25.79, 63.80, 3.650, 53.9, 326.0),
}


def _interp_table(P: float):
    Ps = sorted(_PH2_TABLE)
    P = min(max(P, Ps[0]), Ps[-1])
    cols = np.array([_PH2_TABLE[p] for p in Ps])      # (n,5)
    out = [np.interp(P, Ps, cols[:, j]) for j in range(5)]
    _, rho_l, rho_g, u_l_kJ, u_g_kJ = out
    return rho_l, rho_g, u_l_kJ * 1e3, u_g_kJ * 1e3   # -> J/kg


def _sat_props(P: float):
    return _sat_props_coolprop(P) if _HAS_COOLPROP else _interp_table(P)


def _u_isochore(P: float, rho: float):
    """
    Specific internal energy [J/kg] of a two-phase H2 mixture at pressure P
    and overall density rho (fixed along the constant-volume isochore).
    """
    if _HAS_COOLPROP:
        # (P, D) uniquely fixes the two-phase state for a pure fluid.
        return PropsSI("U", "P", P, "D", rho, FLUID)
    rho_l, rho_g, u_l, u_g = _interp_table(P)
    v, v_l, v_g = 1.0 / rho, 1.0 / rho_l, 1.0 / rho_g
    x = (v - v_l) / (v_g - v_l)                        # vapor mass fraction
    x = min(max(x, 0.0), 1.0)
    return u_l + x * (u_g - u_l)


def heat_leak_budget(geom: TankGeometry, P_vent: float, y_l0: float,
                     tau_H: float = 24 * 3600, P0: float = 101300.0) -> dict:
    """
    Maximum allowable heat-leak rate set by the standby pressure-rise limit.

    geom    : tank geometry (volume V is used)
    P_vent  : vent pressure [Pa]  (tank pressure must not exceed this)
    y_l0    : initial liquid VOLUME fraction after filling (e.g. 0.97)
    tau_H   : standby period [s]  (paper: 120 min = 7200 s)
    P0      : initial fill pressure [Pa] (~1 atm)

    Returns Q_leak [W] and the supporting state quantities.
    """
    rho_l0, rho_g0, *_ = _sat_props(P0)
    rho = y_l0 * rho_l0 + (1.0 - y_l0) * rho_g0        # fixed overall density
    m = rho * geom.V                                   # total H2 mass [kg]

    u_i = _u_isochore(P0, rho)
    u_f = _u_isochore(P_vent, rho)
    dU = m * (u_f - u_i)                                # heat absorbed [J]
    Q_leak = dU / tau_H                                 # [W]
    return {"Q_leak": Q_leak, "m_H2": m, "rho": rho, "dU": dU,
            "u_i": u_i, "u_f": u_f}


# =====================================================================
# 3. SUB-MODEL:  air natural convection
# =====================================================================
# --- Air properties (film temperature) and Churchill-Chu, Ref [35] -----
def _air_props(Tf: float):
    """Simple correlations for dry air at film temperature Tf [K], ~1 atm."""
    k = 0.0241 * (Tf / 293.0) ** 0.9                    # W/(m.K), ~0.026 @300K
    nu = 1.46e-5 * (Tf / 293.0) ** 1.75                 # kinematic visc [m^2/s]
    alpha = 2.05e-5 * (Tf / 293.0) ** 1.75              # thermal diff [m^2/s]
    Pr = nu / alpha
    return k, nu, alpha, Pr


def h_air_horizontal_cylinder(T_s: float, geom: TankGeometry,
                              T0: float = 293.0) -> float:
    """
    Natural-convection coefficient [W/(m^2.K)] on a horizontally oriented
    tank, Churchill-Chu correlation for a horizontal cylinder. The
    characteristic length is the tank outer diameter D = 2*max(a, c).
    """
    D = 2.0 * max(geom.a, geom.c)
    Tf = 0.5 * (T0 + T_s)
    k, nu, alpha, Pr = _air_props(Tf)
    beta = 1.0 / Tf
    g = 9.81
    dT = max(T0 - T_s, 1e-6)
    Ra = g * beta * dT * D ** 3 / (nu * alpha)
    Nu = (0.60 + 0.387 * Ra ** (1.0 / 6.0) /
          (1.0 + (0.559 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    return Nu * k / D


def solve_Ts_for_Q(Q: float, geom: TankGeometry, T0: float = 293.0) -> float:
    """
    Solve the convection balance  Q = h_air(Ts) * A * (T0 - Ts)  for Ts,
    by bisection on Ts in (20 K, T0).
    """
    A = geom.A_out
    f = lambda Ts: h_air_horizontal_cylinder(Ts, geom, T0) * A * (T0 - Ts) - Q
    lo, hi = 20.0, T0 - 1e-4
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:                       # Q outside achievable range
        return hi if abs(fhi) < abs(flo) else lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-9 or (hi - lo) < 1e-8:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


# =====================================================================
# 4. MAIN SOLVER  (MLI vacuum-jacket sizing via the Lockheed equation)
# =====================================================================
@dataclass
class MLIResult:
    N_layers: int           # number of reflector layers (rounded up)
    thickness: float        # blanket thickness [m]  (= N / Nbar)
    flux: float             # achieved heat flux [W/m^2]
    Q_leak: float           # total heat leak q*A [W]
    Q_target: float         # allowable heat leak from pressure budget [W]
    T_s: float              # outer (vacuum-jacket) surface temperature [K]
    A: float                # tank surface area [m^2]
    m_H2: float             # hydrogen mass [kg]
 
 
def lockheed_flux(N: float, T_h: float, T_c: float, Nbar: float,
                  eps: float = 0.043, P_torr: float = 1e-5) -> float:
    """
    Heat flux [W/m^2] through an N-layer MLI blanket (Lockheed equation).
 
    N     : number of reflector layers
    T_h   : warm-boundary temperature [K]
    T_c   : cold-boundary temperature [K]
    Nbar  : layer density [layers/cm]
    eps   : reflector emissivity (DAM ~ 0.043)
    P_torr: residual gas pressure in the vacuum space [torr]
    """
    C_S = 8.95e-8
    C_R = 5.39e-10
    C_G = 1.46e-4

    T_m = 0.5 * (T_h + T_c)
    q_solid = C_S * T_m * Nbar ** 2.63 * (T_h - T_c) / (N - 1.0)
    q_rad = C_R * eps * (T_h ** 4.67 - T_c ** 4.67) / N
    q_gas = C_G * P_torr * (T_h ** 0.52 - T_c ** 0.52) / N
    return q_solid + q_rad + q_gas
 
 
def _solve_layers_for_flux(q_target: float, T_h: float, T_c: float, Nbar: float,
                           eps: float, P_torr: float) -> float:
    """
    Solve the Lockheed equation q(N) = q_target for the (real) layer count N.
 
    q(N) = a/(N-1) + b/N  with
        a = C_S*T_m*Nbar^2.63*(T_h-T_c)              (solid term)
        b = C_R*eps*(Th^4.67-Tc^4.67)
          + C_G*P*(Th^0.52-Tc^0.52)                  (radiation + gas)
    which rearranges to the quadratic
        q_t*N^2 - (q_t + a + b)*N + b = 0   ->  take the larger root.
    """
    C_S = 8.95e-8
    C_R = 5.39e-10
    C_G = 1.46e-4

    T_m = 0.5 * (T_h + T_c)
    a = C_S * T_m * Nbar ** 2.63 * (T_h - T_c)
    b = (C_R * eps * (T_h ** 4.67 - T_c ** 4.67)
         + C_G * P_torr * (T_h ** 0.52 - T_c ** 0.52))
    A = q_target
    B = -(q_target + a + b)
    C = b
    disc = B ** 2 - 4.0 * A * C
    if A <= 0 or disc < 0:
        raise ValueError("No real layer count meets the target flux; "
                         "lower the target or check inputs.")
    N = (-B + math.sqrt(disc)) / (2.0 * A)
    return max(N, 2.0)          # need at least 2 reflector layers
 
 
def mli_thickness(V: float, phi: float, lam: float, P_vent: float,
                  y_l0: float = 0.97, tau_H: float = 2*3600,
                  T_h: float = 293.0, T_c: float = 20.0,
                  Nbar: float = 24.0, eps: float = 0.043,
                  P_torr: float = 1e-5,
                  target_flux: float | None = None) -> MLIResult:
    """
    Size an MLI vacuum-jacket blanket via the Lockheed equation.
 
    By default the blanket is sized to the SAME standby pressure-rise budget
    used for the foam option: q_target = Q_allowable / A. Pass `target_flux`
    [W/m^2] to size to a fixed flux instead.
 
    Parameters
    ----------
    V, phi, lam, P_vent, y_l0, tau_H : as in `insulation_thickness`
    T_h     : warm-boundary (vacuum-jacket) temperature [K]
    T_c     : cold-boundary (tank wall) temperature [K]
    Nbar    : MLI layer density [layers/cm]
    eps     : reflector emissivity
    P_torr  : residual gas pressure in the vacuum space [torr]
    target_flux : optional fixed design flux [W/m^2]; overrides the budget.
    """
    geom = build_tank(V, phi, lam)
    budget = heat_leak_budget(geom, P_vent, y_l0, tau_H)
    Q_target = budget["Q_leak"]
 
    q_target = target_flux if target_flux is not None else Q_target / geom.A_out

    N_real = _solve_layers_for_flux(q_target, T_h, T_c, Nbar, eps, P_torr)
    N = int(math.ceil(N_real))                      # whole layers, round up

    q_achieved = lockheed_flux(N, T_h, T_c, Nbar, eps, P_torr)
    thickness = N / Nbar / 100.0                    # layers/(layers/cm) -> m
    Q_leak = q_achieved * geom.A_out

    # Outer surface temperature from natural convection of ambient air,
    # balancing the achieved heat leak against h_air(Ts)*A*(T0 - Ts).
    T_s = solve_Ts_for_Q(Q_leak, geom, T_h)

    return MLIResult(
        N_layers=N, thickness=thickness, flux=q_achieved,
        Q_leak=Q_leak, Q_target=Q_target, T_s=T_s,
        A=geom.A_out, m_H2=budget["m_H2"])
 
 
# =====================================================================
# Example
# =====================================================================
if __name__ == "__main__":
    # Optimum tank shape from the paper (phi ~ 1, lambda ~ 0.55),
    # P_vent = 1.63 bar, holding 100 m^3 of LH2 at 97% fill.
    V_LH2 = 3.50           # m^3 of liquid hydrogen
    y_l0 = 0.97
    V = V_LH2 / y_l0        # total tank volume

    tau_H = 24 * 3600.0      # standby period [s]  (paper: 120 min)
    Nbar = 30.0             # MLI layer density [layers/cm]

    print(f"CoolProp available : {_HAS_COOLPROP}")

    print("\n--- MLI vacuum jacket (Lockheed equation) ---")
    mli = mli_thickness(V=V, phi=1.0, lam=0.55, P_vent=1.63e5, y_l0=y_l0,
                        tau_H=tau_H, T_h=293.0, T_c=20.0, Nbar=Nbar,
                        eps=0.043, P_torr=1e-5)
    print(f"Tank surface area A: {mli.A:8.2f} m^2")
    print(f"H2 mass            : {mli.m_H2:8.1f} kg")
    print(f"Allowable heat leak: {mli.Q_target:8.1f} W")
    print(f"Number of layers N : {mli.N_layers:8d}")
    print(f"Achieved flux q    : {mli.flux:8.4f} W/m^2")
    print(f"Total heat leak    : {mli.Q_leak:8.1f} W")
    print(f"Surface temp T_s   : {mli.T_s:8.2f} K")
    print(f"Blanket thickness  : {mli.thickness * 1000:8.2f} mm  (at {Nbar} layers/cm)")