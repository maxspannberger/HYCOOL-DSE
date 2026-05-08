"""
Mission_Power.py

Computes shaft power requirements per mission phase and converts them
to LH2 fuel mass via thermal efficiency and hydrogen LHV.

Phases:
    - Cruise:  Breguet-equivalent constant-altitude/Mach segment
    - Reserve: hold at 1500 ft (above destination) at 1.3 V_stall
    - Climb:   average ROC, average power
    - Takeoff: thrust * V_LOF / eta_prop (reference only, not added to fuel)
    - Taxi+TO: lumped allowance as fraction of (cruise + climb) fuel

Drag handling:
    Cruise CD0 and Oswald e from the drag module are reused for hold
    and climb. CD0 does drift with Reynolds (~10% across this range),
    but for Class II that's well within the noise. Induced drag is
    recomputed at the local CL.

Energy chain:
    P_shaft = D * V / eta_prop                      (cruise/hold)
    P_shaft = (D * V + W * ROC) / eta_prop          (climb)
    P_fuel  = P_shaft / eta_thermal
    m_dot   = P_fuel / LHV_LH2
    m_fuel  = m_dot * t_phase
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ISA import isa
from Aircraft_Config   import AircraftConfig
from ClassII_Drag   import DragBreakdown


G        = 9.80665          # m/s^2
LHV_LH2  = 120e6            # J/kg, lower heating value of liquid hydrogen
RHO_SL   = 1.225            # kg/m^3, ISA sea level


@dataclass
class MissionFuelBreakdown:

    # Shaft powers [W]
    P_cruise_shaft:   float = 0.0
    P_reserve_shaft:  float = 0.0
    P_climb_shaft:    float = 0.0
    P_TO_shaft:       float = 0.0       # reference only

    # Fuel mass flows [kg/s]
    mdot_cruise:      float = 0.0
    mdot_reserve:     float = 0.0
    mdot_climb:       float = 0.0

    # Phase durations [s]
    t_cruise:         float = 0.0
    t_reserve:        float = 0.0
    t_climb:          float = 0.0

    # Fuel masses [kg]
    m_LH2_cruise:     float = 0.0
    m_LH2_reserve:    float = 0.0
    m_LH2_climb:      float = 0.0
    m_LH2_TO_taxi:    float = 0.0

    # Local aerodynamic state per phase (for traceability)
    CL_cruise:        float = 0.0
    CL_reserve:       float = 0.0
    CL_climb:         float = 0.0
    LD_cruise:        float = 0.0
    LD_reserve:       float = 0.0
    LD_climb:         float = 0.0
    V_reserve_TAS:    float = 0.0
    V_climb_TAS:      float = 0.0

    @property
    def m_LH2_total(self) -> float:
        return (self.m_LH2_cruise + self.m_LH2_reserve
                + self.m_LH2_climb + self.m_LH2_TO_taxi)

    @property
    def P_max(self) -> float:
        """Largest required shaft power across phases (excludes TO)."""
        return max(self.P_cruise_shaft, self.P_reserve_shaft, self.P_climb_shaft)

    def summary(self) -> str:
        def kw(p): return p / 1000.0
        lines = [
            "=" * 60,
            "  Mission Power & Fuel Breakdown  (LH2)",
            "=" * 60,
            f"  Phase         Power [kW]   m_dot [g/s]    Time [s]   Fuel [kg]",
            "  " + "-" * 58,
            f"  Cruise        {kw(self.P_cruise_shaft):>9.1f}    {self.mdot_cruise*1000:>9.2f}   {self.t_cruise:>8.0f}   {self.m_LH2_cruise:>8.2f}",
            f"  Climb         {kw(self.P_climb_shaft):>9.1f}    {self.mdot_climb*1000:>9.2f}   {self.t_climb:>8.0f}   {self.m_LH2_climb:>8.2f}",
            f"  Reserve       {kw(self.P_reserve_shaft):>9.1f}    {self.mdot_reserve*1000:>9.2f}   {self.t_reserve:>8.0f}   {self.m_LH2_reserve:>8.2f}",
            f"  TO+Taxi       {'-':>9}    {'-':>9}   {'-':>8}   {self.m_LH2_TO_taxi:>8.2f}",
            "  " + "-" * 58,
            f"  Total LH2 fuel:                                    {self.m_LH2_total:>8.2f} kg",
            "  " + "-" * 58,
            f"  P_TO_shaft (reference): {kw(self.P_TO_shaft):.1f} kW",
            f"  P_max     (cruise/climb/reserve): {kw(self.P_max):.1f} kW",
            "  " + "-" * 58,
            f"  Aerodynamic state:",
            f"    cruise:   CL = {self.CL_cruise:.3f}   L/D = {self.LD_cruise:.2f}",
            f"    climb:    CL = {self.CL_climb:.3f}   L/D = {self.LD_climb:.2f}   V = {self.V_climb_TAS:.1f} m/s TAS",
            f"    reserve:  CL = {self.CL_reserve:.3f}   L/D = {self.LD_reserve:.2f}   V = {self.V_reserve_TAS:.1f} m/s TAS",
            "=" * 60,
        ]
        return "\n".join(lines)


class MissionPower:
    """
    Couples cruise drag (CD0, Oswald e) from the drag module with the
    aircraft config to produce phase-by-phase shaft power and LH2 fuel
    mass. Designed to be called inside the MTOW iteration loop.
    """

    def __init__(
        self,
        cfg:        AircraftConfig,
        drag_bd:    DragBreakdown,
        MTOW:       float,
    ):
        self.cfg     = cfg
        self.drag    = drag_bd
        self.MTOW    = MTOW

        # Cache drag-polar coefficients from cruise drag breakdown
        self.CD0  = drag_bd.CD0 + drag_bd.CD_wave   # zero-lift incl. wave
        self.e    = drag_bd.e

    # ---------- helpers --------------------------------------------------

    def _polar_CD(self, CL: float) -> float:
        """Quadratic drag polar at the cached CD0 and e."""
        return self.CD0 + CL**2 / (np.pi * self.cfg.AR * self.e)

    def _eas_to_tas(self, V_eas: float, rho: float) -> float:
        return V_eas * np.sqrt(RHO_SL / rho)

    def _shaft_power_level(self, W: float, V: float, rho: float) -> tuple[float, float, float]:
        """
        Steady level flight (climb rate = 0) shaft power.
        Returns (P_shaft, CL, L/D).
        """
        q  = 0.5 * rho * V**2
        CL = W / (q * self.cfg.S_ref)
        CD = self._polar_CD(CL)
        D  = q * self.cfg.S_ref * CD
        P  = D * V / self.cfg.eta_prop
        LD = CL / CD
        return P, CL, LD

    def _shaft_power_climb(
        self, W: float, V: float, rho: float, ROC: float,
    ) -> tuple[float, float, float]:
        """
        Steady-climb shaft power.  P_prop * eta_prop = D*V + W*ROC.
        Returns (P_shaft, CL, L/D).
        """
        q  = 0.5 * rho * V**2
        CL = W / (q * self.cfg.S_ref)        # approx (cos(gamma)~1 for small angles)
        CD = self._polar_CD(CL)
        D  = q * self.cfg.S_ref * CD
        P  = (D * V + W * ROC) / self.cfg.eta_prop
        LD = CL / CD
        return P, CL, LD

    def _mdot(self, P_shaft: float) -> float:
        """Convert shaft power to fuel mass flow."""
        P_fuel = P_shaft / self.cfg.eta_thermal
        return P_fuel / self.cfg.LHV_fuel

    # ---------- per-phase ------------------------------------------------

    def _cruise(self) -> tuple[float, float, float, float, float, float]:
        cfg = self.cfg
        T, _, rho = isa(cfg.altitude_cruise)
        V         = cfg.V_cruise
        W_cruise  = 0.95 * self.MTOW * G

        P, CL, LD = self._shaft_power_level(W_cruise, V, rho)
        mdot      = self._mdot(P)
        t         = cfg.range_m / V
        m         = mdot * t
        return P, mdot, t, m, CL, LD

    def _reserve(self) -> tuple[float, float, float, float, float, float, float]:
        cfg       = self.cfg
        T, _, rho = isa(cfg.altitude_reserve)
        V_eas     = 1.3 * cfg.V_stall
        V_tas     = self._eas_to_tas(V_eas, rho)

        # Use MZFW estimate at reserve start (post-cruise weight)
        # Conservative: use MZFW = MTOW - cruise fuel; for Class II we
        # don't yet know cruise fuel exactly, so use MZFW ~ MTOW - small.
        # The standard approach: reserve evaluated at landing weight,
        # which is OEW + payload + reserve fuel. For Class II we
        # approximate with MZFW = MTOW (slight conservatism since
        # fuel mass is small for LH2).
        W_reserve = self.MTOW * G        # conservative

        P, CL, LD = self._shaft_power_level(W_reserve, V_tas, rho)
        mdot      = self._mdot(P)
        t         = cfg.t_reserve
        m         = mdot * t
        return P, mdot, t, m, CL, LD, V_tas

    def _climb(self) -> tuple[float, float, float, float, float, float, float]:
        cfg = self.cfg

        # Evaluate at midpoint altitude
        h_mid     = 0.5 * cfg.altitude_cruise
        T, _, rho = isa(h_mid)
        V_tas     = self._eas_to_tas(cfg.V_climb_EAS, rho)
        ROC       = cfg.ROC_avg
        W_climb   = 0.99 * self.MTOW * G    # small fuel burn before climb done

        P, CL, LD = self._shaft_power_climb(W_climb, V_tas, rho, ROC)
        mdot      = self._mdot(P)
        t         = cfg.altitude_cruise / ROC
        m         = mdot * t
        return P, mdot, t, m, CL, LD, V_tas

    def _takeoff_reference(self) -> float:
        """Climb-out shaft power for reference (not added to fuel)."""
        cfg    = self.cfg
        V_LOF  = 1.2 * cfg.V_stall                  # liftoff EAS ~ TAS at SL
        T_tot  = 2.0 * cfg.T_TO_per_engine          # both engines, TO
        return T_tot * V_LOF / cfg.eta_prop

    # ---------- top level ------------------------------------------------

    def compute(self) -> MissionFuelBreakdown:

        P_c, md_c, t_c, m_c, CL_c, LD_c                       = self._cruise()
        P_r, md_r, t_r, m_r, CL_r, LD_r, V_r                  = self._reserve()
        P_cl, md_cl, t_cl, m_cl, CL_cl, LD_cl, V_cl           = self._climb()
        P_TO                                                  = self._takeoff_reference()

        m_TO_taxi = self.cfg.TO_taxi_frac * (m_c + m_cl)

        return MissionFuelBreakdown(
            P_cruise_shaft  = P_c,
            P_reserve_shaft = P_r,
            P_climb_shaft   = P_cl,
            P_TO_shaft      = P_TO,
            mdot_cruise     = md_c,
            mdot_reserve    = md_r,
            mdot_climb      = md_cl,
            t_cruise        = t_c,
            t_reserve       = t_r,
            t_climb         = t_cl,
            m_LH2_cruise    = m_c,
            m_LH2_reserve   = m_r,
            m_LH2_climb     = m_cl,
            m_LH2_TO_taxi   = m_TO_taxi,
            CL_cruise       = CL_c,
            CL_reserve      = CL_r,
            CL_climb        = CL_cl,
            LD_cruise       = LD_c,
            LD_reserve      = LD_r,
            LD_climb        = LD_cl,
            V_reserve_TAS   = V_r,
            V_climb_TAS     = V_cl,
        )