"""
Thermal design of the LH2 storage tank:
  - maximum allowable heat leak (Ei / Ef)
  - polyurethane foam insulation sizing (Eq. 2)
  - vacuum / MLI insulation sizing (Lockheed equation)
"""

from CoolProp.CoolProp import PropsSI
from scipy.integrate import quad
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional
from geomDesign import Geometry
import properties as props


FLUID = "parahydrogen"
BAR = 1e5  # Pa per bar


@dataclass(frozen=True)
class InsulationResult:
    delta_ins:   float          # insulation thickness [m]
    m_ins:       float          # insulation mass [kg]
    Q_leak:      float          # effective heat leak [W]
    Ts:          float          # outer surface temperature [K]
    constrained: bool           # True if freeze constraint is active
    N_layers:    Optional[int]  # MLI layer count (vacuum only, else None)

    def print_summary(self):
        w = 54
        print("\n" + "=" * w)
        print(f"{'  INSULATION SIZING RESULTS':^{w}}")
        print("=" * w)
        print(f"  {'Insulation thickness':<28}  {self.delta_ins*100:>8.2f}  cm")
        if self.N_layers is not None:
            print(f"  {'MLI layers':<28}  {self.N_layers:>8}  -")
        print(f"  {'Insulation mass':<28}  {self.m_ins:>8.2f}  kg")
        print(f"  {'Effective heat leak':<28}  {self.Q_leak:>8.2f}  W")
        print(f"  {'Outer surface temperature':<28}  {self.Ts:>8.2f}  K"
              f"  ({self.Ts - 273.15:.1f} deg C)")
        if self.constrained:
            print(f"\n  NOTE: freeze constraint active — Ts pinned at 273 K.")
        print("=" * w + "\n")


def k_foam(T: float) -> float:
    """Thermal conductivity of closed-cell rigid polyurethane foam [W/(m·K)].

    Linear empirical fit valid over 20–300 K (Winnefeld et al. 2018).
    """
    return 1.5e-4 * T + 0.01352


# ------------------------------------------------------------------
# Lockheed MLI equation constants
# Empirical values for double-aluminized mylar (DAM) with Dacron net spacers.
# Calibrated to give ~1–3 W/m² for 10–20 layers between 293 K and 20 K.
# Adjust to match your specific MLI material and construction.
# ------------------------------------------------------------------
_C_S = 2.7e-7    # solid conduction coefficient  [W/(m²·K²)]
_C_R = 6.0e-10   # radiation coefficient          [W/(m²·K^4.67)]
_C_G = 1.0       # gas conduction coefficient     [W/(m²·Pa·K^0.52)]


def q_lockheed(T_h: float, T_c: float, N: int,
               emissivity: float = 0.03,
               P_residual: float = 1e-4,
               C_s: float = _C_S,
               C_R: float = _C_R,
               C_G: float = _C_G) -> float:
    """
    Lockheed equation: total heat flux through an N-layer MLI system [W/m²].

        Q̇/A = ( C_s · T̄_m · N^2.63 · (T_h − T_c) ) / (N − 1)
             + ( C_R · ε · (T_h^4.67 − T_c^4.67) ) / N
             + ( C_G · P · (T_h^0.52 − T_c^0.52) ) / N

    The three terms represent solid conduction through spacers, radiation
    between layers, and residual gas conduction respectively.

    Parameters
    ----------
    T_h        : hot-wall (outer jacket) temperature [K]
    T_c        : cold-wall (tank) temperature [K]
    N          : number of MLI layers [-]
    emissivity : emissivity of MLI layers [-]
    P_residual : residual gas pressure in the vacuum gap [Pa]
    C_s, C_R, C_G : Lockheed equation constants (see module-level defaults)

    Notes
    -----
    N = 0 returns the bare two-wall radiation heat flux (no shields).
    N = 1 clamps the solid-conduction denominator to 1 to avoid division by zero;
    for N ≥ 2 the equation is used as written.
    """
    _SIGMA = 5.670374419e-8
    if N == 0:
        return _SIGMA * (T_h**4 - T_c**4) / (2.0 / emissivity - 1.0)
    T_m    = 0.5 * (T_h + T_c)
    denom  = max(N - 1, 1)          # clamp at N=1 to avoid zero denominator
    q_solid = C_s * T_m * N**2.63 * (T_h - T_c) / denom
    q_rad   = C_R * emissivity * (T_h**4.67 - T_c**4.67) / N
    q_gas   = C_G * P_residual * (T_h**0.52 - T_c**0.52) / N
    return q_solid + q_rad + q_gas


def k_MLI(T: float, N: int,
          t_layer: float = 3e-3,
          emissivity: float = 0.03,
          P_residual: float = 1e-4,
          C_s: float = _C_S,
          C_R: float = _C_R,
          C_G: float = _C_G) -> float:
    """
    Differential effective thermal conductivity of N-layer MLI [W/(m·K)],
    derived from the Lockheed equation:

        k_MLI(T) = N · t_layer · ∂(Q̇/A) / ∂T_h

    where the partial derivative is taken with T_c fixed:

        ∂(Q̇/A)/∂T_h = C_s · N^2.63/(N−1) · T
                      + C_R · ε/N · 4.67 · T^3.67
                      + C_G · P/N · 0.52 · T^−0.48

    Integrating k_MLI from T_c to T_h recovers the Lockheed heat flux,
    consistent with the Eq. (2) framework used for foam insulation.

    Valid for N ≥ 2.
    """
    if N < 2:
        raise ValueError(f"k_MLI (Lockheed) requires N >= 2, got N = {N}.")
    dk_solid = C_s * N**2.63 / (N - 1) * T
    dk_rad   = C_R * emissivity / N * 4.67 * T**3.67
    dk_gas   = C_G * P_residual / N * 0.52 * T**(-0.48)
    return N * t_layer * (dk_solid + dk_rad + dk_gas)


class ThermalDesign:
    def __init__(self):
        pass

    def calculateMaxHeatLeak(self, V_tank: float, yl_0: float,
                             p_fill_bar: float, p_vent_bar: float,
                             y_max: float, tau_H_s: float):
        """
        Calculate the maximum allowable heat leak rate Q_leak (W).

        Params
        ------
        V_tank     : total tank volume [m³]
        yl_0       : initial liquid volume fraction at fill [-]
        p_fill_bar : fill pressure [bar]
        p_vent_bar : vent pressure [bar]
        y_max      : liquid volume fraction at which venting begins (0.97) [-]
        tau_H_s    : standby period before takeoff [s]

        Returns
        -------
        Q_leak : maximum allowable heat leak rate [W]
        Ei     : initial total internal energy of hydrogen [J]
        Ef     : final total internal energy at P_vent [J]
        """
        P0 = p_fill_bar * BAR
        P_vent = p_vent_bar * BAR

        vl_0, vg_0 = props.saturated_specific_volumes(P0, FLUID)
        vl_v, vg_v = props.saturated_specific_volumes(P_vent, FLUID)

        ul_0, ug_0 = props.saturated_internal_energies(P0, FLUID)
        ul_v, ug_v = props.saturated_internal_energies(P_vent, FLUID)

        # Internal energy per unit tank volume [J/m³]:
        # E/V = (y_l / v_l) * u_l + ((1 - y_l) / v_g) * u_g
        ei_density = (yl_0 / vl_0) * ul_0 + ((1 - yl_0) / vg_0) * ug_0
        ef_density = (y_max / vl_v) * ul_v + ((1 - y_max) / vg_v) * ug_v

        Ei = ei_density * V_tank
        Ef = ef_density * V_tank

        Q_leak = (Ef - Ei) / tau_H_s
        return Q_leak, Ei, Ef

    # ------------------------------------------------------------------
    # Shared helper: Churchill-Chu natural convection on horizontal cylinder
    # ------------------------------------------------------------------

    def _h_air(self, Ts: float, D_char: float, T0: float) -> float:
        T_film = 0.5 * (T0 + Ts)
        k_a   = PropsSI('conductivity', 'P', 101325.0, 'T', T_film, 'Air')
        Pr    = PropsSI('Prandtl',      'P', 101325.0, 'T', T_film, 'Air')
        mu_a  = PropsSI('viscosity',    'P', 101325.0, 'T', T_film, 'Air')
        rho_a = PropsSI('D',            'P', 101325.0, 'T', T_film, 'Air')
        nu_a  = mu_a / rho_a
        beta  = 1.0 / T_film
        Ra = 9.81 * beta * abs(T0 - Ts) * D_char**3 * Pr / nu_a**2
        Nu = (0.60 + 0.387 * Ra**(1/6) /
              (1.0 + (0.559 / Pr)**(9/16))**(8/27))**2
        return Nu * k_a / D_char

    # ------------------------------------------------------------------
    # Shared core: Eq. (2) iterative insulation sizing
    # ------------------------------------------------------------------

    def _size_insulation_core(self, geom: Geometry, Q_leak_max: float,
                              k_ins_func, p_fill_bar: float,
                              T0: float, T_freeze: float) -> tuple:
        """
        Iterative solver for Eq. (2):  δ = (A/Q̇) · ∫[T_LH2→Ts] k_ins(T) dT
        combined with external convection: Q̇ = h_air · A · (T0 − Ts).

        Returns (delta, Q_leak, Ts, constrained).
        """
        A = geom.A_tank
        c = geom.c
        T_LH2 = PropsSI('T', 'P', p_fill_bar * BAR, 'Q', 0, FLUID)

        delta = 0.05
        constrained = False
        Q_leak = Q_leak_max
        Ts = T0 - 10.0

        for _ in range(200):
            D_outer = 2.0 * (c + delta)
            Q_at_freeze = self._h_air(T_freeze, D_outer, T0) * A * (T0 - T_freeze)

            if Q_at_freeze < Q_leak_max:
                Ts = T_freeze
                Q_leak = Q_at_freeze
                constrained = True
            else:
                Ts = brentq(
                    lambda Ts_: self._h_air(Ts_, D_outer, T0) * A * (T0 - Ts_) - Q_leak_max,
                    T_freeze + 1e-3, T0 - 1e-3
                )
                Q_leak = Q_leak_max
                constrained = False

            integral, _ = quad(k_ins_func, T_LH2, Ts)
            delta_new = (A / Q_leak) * integral

            if abs(delta_new - delta) < 1e-7:
                delta = delta_new
                break

            delta = 0.5 * (delta + delta_new)

        return delta, Q_leak, Ts, constrained

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def sizeInsulation(self, geom: Geometry, Q_leak_max: float,
                       p_fill_bar: float = 1.0,
                       T0: float = 293.0, T_freeze: float = 273.0,
                       rho_ins: float = 40.0) -> dict:
        """
        Size the polyurethane foam insulation layer using Eq. (2).

            δ_ins = (A / Q̇_leak) · ∫[T_LH2 → Ts] k_foam(T) dT

        Ts is found from external natural-convection (Churchill-Chu, horizontal
        cylinder). If Ts < T_freeze the constraint is activated and Q̇_leak is
        reduced to h_air · A · (T0 − T_freeze).

        Parameters
        ----------
        geom        : Geometry dataclass from geomDesign
        Q_leak_max  : maximum allowable heat leak rate [W]
        p_fill_bar  : fill pressure [bar]
        T0          : ambient temperature [K]
        T_freeze    : minimum allowed outer-surface temperature [K]
        rho_ins     : polyurethane foam density [kg/m³]

        Returns
        -------
        InsulationResult
        """
        delta, Q_leak, Ts, constrained = self._size_insulation_core(
            geom, Q_leak_max, k_foam, p_fill_bar, T0, T_freeze
        )
        return InsulationResult(
            delta_ins   = delta,
            m_ins       = rho_ins * geom.A_tank * delta,
            Q_leak      = Q_leak,
            Ts          = Ts,
            constrained = constrained,
            N_layers    = None,
        )

    def sizeVacuumInsulation(self, geom: Geometry, Q_leak_max: float,
                             p_fill_bar: float = 1.0,
                             T0: float = 293.0, T_freeze: float = 273.0,
                             emissivity: float = 0.03,
                             t_layer: float = 3e-3,
                             rho_layer_areal: float = 0.20,
                             delta_gap: float = 0.05,
                             P_residual: float = 1e-4,
                             C_s: float = _C_S,
                             C_R: float = _C_R,
                             C_G: float = _C_G) -> InsulationResult:
        """
        Size MLI for a vacuum-jacketed tank using the Lockheed equation.

        The outer-surface temperature Ts is found from the Churchill-Chu
        external natural-convection balance (same as sizeInsulation). The
        minimum integer N such that:

            q_lockheed(Ts, T_LH2, N) · A ≤ Q̇_leak

        is then found by incrementing N from 0. Since D_outer = 2(c + δ_gap + N·t_layer)
        feeds back into h_air → Ts, the procedure iterates until N converges.

        Parameters
        ----------
        geom             : Geometry dataclass
        Q_leak_max       : maximum allowable heat leak rate [W]
        p_fill_bar       : fill pressure [bar]
        T0               : ambient temperature [K]
        T_freeze         : minimum allowed outer-surface temperature [K]
        emissivity       : emissivity of MLI layers and tank walls [-]
        t_layer          : thickness per MLI layer including spacers [m]
        rho_layer_areal  : areal mass density per MLI layer [kg/m²]
        delta_gap        : minimum structural vacuum gap (independent of N) [m]
        P_residual       : residual gas pressure in the vacuum gap [Pa]
        C_s, C_R, C_G   : Lockheed equation constants (see module-level defaults)

        Returns
        -------
        InsulationResult
        """
        A   = geom.A_tank
        c   = geom.c
        T_LH2 = PropsSI('T', 'P', p_fill_bar * BAR, 'Q', 0, FLUID)

        N           = 0
        constrained = False
        Q_leak      = Q_leak_max
        Ts          = T0 - 10.0

        for _ in range(50):
            D_outer      = 2.0 * (c + delta_gap + N * t_layer)
            Q_at_freeze  = self._h_air(T_freeze, D_outer, T0) * A * (T0 - T_freeze)

            if Q_at_freeze < Q_leak_max:
                Ts          = T_freeze
                Q_leak      = Q_at_freeze
                constrained = True
            else:
                Ts = brentq(
                    lambda Ts_: self._h_air(Ts_, D_outer, T0) * A * (T0 - Ts_) - Q_leak_max,
                    T_freeze + 1e-3, T0 - 1e-3,
                )
                Q_leak      = Q_leak_max
                constrained = False

            q_target = Q_leak / A

            # Find minimum N such that Lockheed heat flux ≤ target
            N_new = next(
                (n for n in range(200)
                 if q_lockheed(Ts, T_LH2, n, emissivity, P_residual, C_s, C_R, C_G) <= q_target),
                200,
            )

            if N_new == N:
                break
            N = N_new

        Q_leak_actual = q_lockheed(Ts, T_LH2, N, emissivity, P_residual, C_s, C_R, C_G) * A

        return InsulationResult(
            delta_ins   = delta_gap + N * t_layer,
            m_ins       = rho_layer_areal * A * N,
            Q_leak      = Q_leak_actual,
            Ts          = Ts,
            constrained = constrained,
            N_layers    = N,
        )


def _print_results(ins_type: str, Q_leak_max: float,
                   result: InsulationResult) -> None:
    print(f"\n  Insulation type : {ins_type.upper()}")
    print(f"  Q_leak_max      : {Q_leak_max:.2f} W")
    result.print_summary()


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  USER INPUTS                                                         #
    # ------------------------------------------------------------------ #
    insulation_type = 'vacuum'     # 'foam' or 'vacuum'
    tau_H_hours     = 24.0        # holding time [hours]

    m_H2    = 500.0              # hydrogen mass per tank [kg]
    p_fill  = 1.0                # fill pressure [bar]
    p_vent  = 1.63               # vent pressure [bar]
    y_max   = 0.97               # max liquid volume fraction before venting [-]
    phi     = 1.0                # tank shape: a/c ratio
    psi     = 1.0                # tank shape: b/c ratio
    Lambda  = 0.55               # tank shape: shell fraction
    # ------------------------------------------------------------------ #

    from geomDesign import GeomDesign

    tau_H_s = tau_H_hours * 3600.0

    # 1. Geometry
    gd   = GeomDesign(p_vent=p_vent, p_fill=p_fill, y_max=y_max)
    yl_0 = gd.calculateInitialLiquidMassFraction(yl_vent=y_max)
    rho  = PropsSI('D', 'P', p_vent * BAR, 'Q', 0, FLUID)
    V    = gd.calculateTankVolume(rho_H2=rho, m_H2=m_H2, yl_0=yl_0)
    geom = gd.calculateTankGeometry(V, phi=phi, psi=psi, Lambda=Lambda)

    w = 54
    print("\n" + "=" * w)
    print(f"{'  DESIGN INPUTS':^{w}}")
    print("=" * w)
    print(f"  {'Insulation type':<28}  {insulation_type.upper():>10}")
    print(f"  {'Holding time':<28}  {tau_H_hours:>10.1f}  h")
    print(f"  {'Hydrogen mass':<28}  {m_H2:>10.1f}  kg")
    print(f"  {'Fill pressure':<28}  {p_fill:>10.2f}  bar")
    print(f"  {'Vent pressure':<28}  {p_vent:>10.2f}  bar")
    print(f"  {'Initial liquid fraction':<28}  {yl_0:>10.4f}  -")
    print(f"  {'Tank volume':<28}  {V:>10.4f}  m^3")
    print(f"  {'Tank surface area':<28}  {geom.A_tank:>10.4f}  m^2")
    print("=" * w)

    # 2. Max heat leak
    td = ThermalDesign()
    Q_leak, Ei, Ef = td.calculateMaxHeatLeak(V, yl_0, p_fill, p_vent, y_max, tau_H_s)

    print(f"\n  Ei          = {Ei/1e6:.4f} MJ")
    print(f"  Ef          = {Ef/1e6:.4f} MJ")
    print(f"  Ef - Ei     = {(Ef-Ei)/1e6:.4f} MJ")
    print(f"  Q_leak_max  = {Q_leak:.2f} W")

    # 3. Size insulation
    if insulation_type == 'foam':
        result = td.sizeInsulation(geom, Q_leak)
    elif insulation_type == 'vacuum':
        result = td.sizeVacuumInsulation(geom, Q_leak)
    else:
        raise ValueError(f"Unknown insulation type '{insulation_type}'. "
                         f"Choose 'foam' or 'vacuum'.")

    _print_results(insulation_type, Q_leak, result)
