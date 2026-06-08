"""
Sizes the tank and calculates the dimensions of the tank. 
"""

from CoolProp.CoolProp import PropsSI
from scipy.integrate import quad
from scipy.optimize import brentq
import numpy as np
from dataclasses import dataclass
from geomDesign import Geometry
import properties as props

"""
@dataclass(frozen=True)
class Thermals:
"""  


FLUID = "parahydrogen"
BAR = 1e5  # Pa per bar


def k_foam(T: float) -> float:
    """Thermal conductivity of closed-cell rigid polyurethane foam [W/(m·K)].

    Linear empirical fit valid over 20–300 K (Winnefeld et al. 2018).
    """
    return 1.5e-4 * T + 0.01352


def k_MLI(T: float, t_layer: float = 3e-3, emissivity: float = 0.03) -> float:
    """Effective thermal conductivity per unit thickness of vacuum MLI [W/(m·K)].

    Derived from the differential radiation law between parallel shields (T³ law):
        k_MLI = 4σT³ · t_layer / (2/ε − 1)

    Integrating this in Eq. (2) is equivalent to the standard N-shield radiation
    equation: Q = σA(T_h⁴ − T_c⁴) / [(N+1)(2/ε−1)], consistent with the paper's
    framework but for MLI instead of foam.
    """
    return 4.0 * 5.670374419e-8 * T**3 * t_layer / (2.0 / emissivity - 1.0)


class ThermalDesign:
    def __init__(self):
        pass

    ### Helper functions for thermal design ###

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

    def calculatePrandtlNumber(self, P, T):
        """Calculate the Prandtl number."""
        Pr = PropsSI('Prandtl', 'P', P, 'T', T, 'Hydrogen')
        return Pr
    
    def calculateRayleighNumber(self, P, T1, T2, L, Pr):
        """Calculate the Rayleigh number."""
        beta = PropsSI('isobaric_expansion_coeff', 'P', P, 'T', T1, 'Hydrogen')
        nu = PropsSI('viscosity', 'P', P, 'T', T1, 'Hydrogen') / PropsSI('D', 'P', P, 'T', T1, 'Hydrogen')
        alpha = PropsSI('conductivity', 'P', P, 'T', T1, 'Hydrogen') / (PropsSI('D', 'P', P, 'T', T1, 'Hydrogen') * PropsSI('Cp0', 'P', P, 'T', T1, 'Hydrogen'))
        g = 9.81  # m/s^2
        Ra = (g * beta * (T1 - T2) * (L**3) * Pr) / (nu ** 2)
        return Ra
    
    def calculateHeatTransferCoefficient(self, Nu, k, L):
        """Calculate the heat transfer coefficient."""
        h = (Nu * k) / L
        return h

    ### Main functions for thermal design ###

    def internalConvection(self, geom: Geometry, P, T):
        """Calculate the internal thermal resistance due to natural convection within the tank."""
        R = geom.b
        hf = geom.h_fill
        hv = geom.h_vent

        # 1. Calculate the Nusselt Number and heat transfer coefficient at the tank ceiling.
        Nu_1 = 1.0  # Example calculation; replace with actual calculation
        k_1 = PropsSI('conductivity', 'P', P, 'T', T, 'Hydrogen')
        h_1 = self.calculateHeatTransferCoefficient(Nu_1, k_1, L = 2 * R - hf)  # Example calculation; replace with actual calculation

        # 2. Calculate the Nusselt Number and heat transfer coefficient at the tank floor.
        Pr_2 = self.calculatePrandtlNumber(self, P, T)  # Example calculation; replace with actual calculation
        Ra_2 = self.calculateRayleighNumber(self, P, T1, T2, L, Pr)  # Example calculation; replace with actual calculation
        k_2 = PropsSI('conductivity', 'P', P, 'T', T, 'Hydrogen')

        if Ra_2 < 1e9 and Ra_2 > 1e4:
            Nu_2 = 0.54 * (Ra_2 ** (1/3))  # Example calculation; replace with actual calculation
        elif Ra_2 >= 1e9 and Ra_2 < 1e12:
            Nu_2 = 0.098 * (Ra_2 ** (1/4))  # Example calculation; replace with actual calculation
        
        h_2 = self.calculateHeatTransferCoefficient(Nu_2, k_2, L = hf)  # Example calculation; replace with actual calculation

        # 3. Calculate the Nusselt Number and heat transfer coefficient at the caps in contact with LH2.
        Pr_3 = self.calculatePrandtlNumber(self, P, T)  # Example calculation; replace with actual calculation
        Ra_3 = self.calculateRayleighNumber(P, T1, T2, L, Pr)  # Example calculation; replace with actual calculation
        Nu_3 = (0.825 + ((0.387 * (Ra_3 ** (1/6)))/((1 + (0.492 / Pr_3) ** (9/16)) ** (8/27)))) ** 2  # Example calculation; replace with actual calculation
        k_3 = PropsSI('conductivity', 'P', P, 'T', T, 'Hydrogen')
        h_3 = self.calculateHeatTransferCoefficient(Nu_3, k_3, L = hf)  # Example calculation; replace with actual calculation

        # 4. Calculate the Nusselt Number and heat transfer coefficient at the caps in contact with GH2.
        Pr_4 = self.calculatePrandtlNumber(self, P, T)  # Example calculation; replace with actual calculation
        Ra_4 = self.calculateRayleighNumber(self, P, T1, T2, L, Pr)  # Example calculation; replace with actual calculation
        Nu_4 = (0.825 + ((0.387 * (Ra_4 ** (1/6)))/((1 + (0.492 / Pr_4) ** (9/16)) ** (8/27)))) ** 2
        k_4 = PropsSI('conductivity', 'P', P, 'T', T, 'Hydrogen')
        h_4 = self.calculateHeatTransferCoefficient(Nu_4, k_4, L = 2 * R - hf)  # Example calculation; replace with actual calculation

        # 5. Calculate the overall thermal resistance for internal convection.
        h_in_l = (1 / geom.Sw_fill['total LH2']) * (h_2 * geom.Sw_fill['floor LH2'] + h_3 * geom.Sw_fill['caps LH2'])
        h_in_g = (1 / geom.Sw_fill['total GH2']) * (h_1 * geom.Sw_fill['ceil GH2'] + h_4 * geom.Sw_fill['caps GH2'])

        h_in = (1 / geom.Sw_fill['total']) * (h_in_l * geom.Sw_fill['total LH2'] + h_in_g * geom.Sw_fill['total GH2'])  # Example calculation; replace with actual calculation
        R_in = 1 / (h_in * geom.A_tank)

        return R_in 


    def thermalConduction(self):
        return 
        

    def externalConvection(self):
        return

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
        dict : delta_ins [m], Q_leak [W], Ts [K], m_ins [kg], constrained [bool]
        """
        delta, Q_leak, Ts, constrained = self._size_insulation_core(
            geom, Q_leak_max, k_foam, p_fill_bar, T0, T_freeze
        )
        return {
            'delta_ins':   delta,
            'Q_leak':      Q_leak,
            'Ts':          Ts,
            'm_ins':       rho_ins * geom.A_tank * delta,
            'constrained': constrained,
        }

    def sizeVacuumInsulation(self, geom: Geometry, Q_leak_max: float,
                             p_fill_bar: float = 1.0,
                             T0: float = 293.0, T_freeze: float = 273.0,
                             emissivity: float = 0.03,
                             t_layer: float = 3e-3,
                             rho_layer_areal: float = 0.20,
                             delta_gap: float = 0.05) -> dict:
        """
        Size MLI for a vacuum-jacketed tank following the paper's Eq. (2) framework.

        The effective thermal conductivity of vacuum MLI is derived from the
        differential radiation law for parallel shields (T³ law):

            k_MLI(T) = 4σT³ · t_layer / (2/ε − 1)

        Plugged into Eq. (2):

            δ_MLI = (A / Q̇_leak) · ∫[T_LH2 → Ts] k_MLI(T) dT

        The continuous δ_MLI is then rounded up to an integer layer count
        N = ⌈δ_MLI / t_layer⌉, and Q̇_leak is recalculated for that N using
        the exact radiation equation. Ts and the T_freeze constraint follow
        the same external-convection model as sizeInsulation.

        Parameters
        ----------
        geom             : Geometry dataclass
        Q_leak_max       : maximum allowable heat leak rate [W]
        p_fill_bar       : fill pressure [bar]
        T0               : ambient temperature [K]
        T_freeze         : minimum allowed outer-surface temperature [K]
        emissivity       : emissivity of each MLI layer and walls (aluminized ≈ 0.03)
        t_layer          : thickness per MLI layer incl. spacers [m]
        rho_layer_areal  : areal mass density per MLI layer [kg/m²]
        delta_gap        : minimum structural vacuum gap (independent of N) [m]

        Returns
        -------
        dict : N_layers [-], delta_ins [m], Q_leak [W], Ts [K],
               m_ins [kg], constrained [bool]
        """
        import math

        def _k_mli(T):
            return k_MLI(T, t_layer=t_layer, emissivity=emissivity)

        delta_cont, _, Ts, constrained = self._size_insulation_core(
            geom, Q_leak_max, _k_mli, p_fill_bar, T0, T_freeze
        )

        N = math.ceil(delta_cont / t_layer)

        # Exact heat leak with integer N (radiation equation)
        SIGMA = 5.670374419e-8
        T_LH2 = PropsSI('T', 'P', p_fill_bar * BAR, 'Q', 0, FLUID)
        Q_leak_actual = (SIGMA * geom.A_tank * (Ts**4 - T_LH2**4)
                         / ((N + 1) * (2.0 / emissivity - 1.0)))

        return {
            'N_layers':    N,
            'delta_ins':   delta_gap + N * t_layer,
            'Q_leak':      Q_leak_actual,
            'Ts':          Ts,
            'm_ins':       rho_layer_areal * geom.A_tank * N,
            'constrained': constrained,
        }

