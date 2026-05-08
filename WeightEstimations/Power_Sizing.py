"""
Power_Sizing.py

Sizes required takeoff shaft power and computes static thrust for a
propfan-class powerplant.

P_TO is the larger of:
    (1) P_climb           : peak shaft power from the mission breakdown
                            (usually drives engine sizing for short range)
    (2) P_CS_25_121       : shaft power required to meet the CS-25.121
                            second-segment climb gradient at V_2 with
                            one engine inoperative.

For each engine (assumed identical), static thrust is then estimated from
actuator-disk momentum theory:

    T_static = ( 2 * rho_SL * A_disk * P_per_engine^2 )^(1/3) * eta_static_loss

This static thrust is the right number to feed back into the tail-sizing
OEI rudder check: it is conservative because thrust at V_MC is lower than
thrust at V = 0, so the asymmetric yaw moment used for rudder sizing
ends up slightly oversized rather than undersized.

CS-25.121 second-segment requirement:
    Steady climb, OEI, gear up, takeoff flaps, V_2.
    Required gradient gamma_min = 0.024 for twin, 0.027 for tri, 0.030 for quad.

    Force balance (small angle):
        T_per_eng - D_TO = W * gamma_min        (at V_2, one engine working)
        T_per_eng = W * (1/(L/D)_TO + gamma_min)

    Equivalent shaft power needed by that one working engine:
        P_per_eng = T_per_eng * V_2 / eta_prop_V2

    Total aircraft P_TO requirement = N_engines * P_per_eng
    (each engine must be capable of producing this in case the *other*
     one fails)
"""

import numpy as np
from dataclasses import dataclass

from Aircraft_Config import AircraftConfig
from Mission_Power   import MissionFuelBreakdown


G        = 9.80665
RHO_SL   = 1.225


@dataclass
class PowerSizingBreakdown:

    # Required shaft powers
    P_TO_total:        float = 0.0     # Whole aircraft [W]
    P_TO_per_engine:   float = 0.0     # Per engine [W]

    # Sources
    P_from_climb:      float = 0.0     # Mission peak power
    P_from_CS25_121:   float = 0.0     # Second-segment requirement
    driving_case:      str   = ""      # "climb" or "CS-25.121"

    # CS-25.121 intermediate values
    T_per_engine_V2:   float = 0.0     # Required thrust per engine at V_2
    V_2:               float = 0.0     # CS-25 reference climb speed
    LD_takeoff:        float = 0.0     # L/D in TO config used in calc
    gamma_min:         float = 0.0     # Minimum gradient

    # Static thrust output
    T_static_total:    float = 0.0     # Whole aircraft static thrust [N]
    T_static_per_engine: float = 0.0   # [N]

    def summary(self) -> str:
        kw = lambda p: p / 1000.0
        kn = lambda t: t / 1000.0
        lines = [
            "=" * 60,
            "  Power & Takeoff Thrust Sizing",
            "=" * 60,
            f"  Driving case:           {self.driving_case}",
            f"  P_climb (mission):      {kw(self.P_from_climb):>10.1f} kW",
            f"  P_CS-25.121 (2nd seg):  {kw(self.P_from_CS25_121):>10.1f} kW",
            "  " + "-" * 58,
            f"  Required P_TO total:    {kw(self.P_TO_total):>10.1f} kW",
            f"  Required P_TO/engine:   {kw(self.P_TO_per_engine):>10.1f} kW",
            "  " + "-" * 58,
            f"  CS-25.121 details:",
            f"    V_2                   {self.V_2:>10.2f} m/s",
            f"    gamma_min             {self.gamma_min*100:>10.2f} %",
            f"    (L/D)_TO              {self.LD_takeoff:>10.2f}",
            f"    T per engine at V_2   {kn(self.T_per_engine_V2):>10.2f} kN",
            "  " + "-" * 58,
            f"  Static thrust total:    {kn(self.T_static_total):>10.2f} kN",
            f"  Static thrust per eng:  {kn(self.T_static_per_engine):>10.2f} kN",
            "=" * 60,
        ]
        return "\n".join(lines)


class PowerSizing:

    def __init__(
        self,
        cfg:        AircraftConfig,
        mission:    MissionFuelBreakdown,
        MTOW:       float,
    ):
        self.cfg     = cfg
        self.mission = mission
        self.MTOW    = MTOW

    def _cs25_121_power(self) -> tuple[float, float, float, float]:
        """
        Required *total* shaft power to satisfy CS-25.121 second segment.

        Returns (P_total_required, T_per_engine_at_V2, V_2, gamma_min).
        """
        cfg = self.cfg

        gamma_min_lookup = {2: 0.024, 3: 0.027, 4: 0.030}
        gamma_min        = gamma_min_lookup.get(cfg.N_engines, 0.024)

        # V_2 = 1.2 V_stall (TO config, EAS ~ TAS at sea level)
        V_2 = 1.2 * cfg.V_stall

        # Working-engine thrust requirement at V_2 (per CS-25.121)
        W = self.MTOW * G
        T_per_engine_V2 = W * (1.0 / cfg.LD_takeoff + gamma_min)

        # Convert to shaft power for the one working engine
        P_per_engine = T_per_engine_V2 * V_2 / cfg.eta_prop_V2

        # Total aircraft power = N_engines * per-engine capability,
        # since every engine must be sized to handle the failure case
        P_total = cfg.N_engines * P_per_engine

        return P_total, T_per_engine_V2, V_2, gamma_min

    def _static_thrust(self, P_per_engine: float) -> float:
        """
        Actuator-disk static thrust, with empirical loss factor.
        T = ( 2 * rho_SL * A_disk * P^2 )^(1/3) * eta_static_loss
        """
        cfg     = self.cfg
        A_disk  = np.pi * cfg.D_propfan**2 / 4.0
        T_ideal = (2.0 * RHO_SL * A_disk * P_per_engine**2) ** (1.0 / 3.0)
        return T_ideal * cfg.eta_static_loss

    def compute(self) -> PowerSizingBreakdown:
        cfg = self.cfg

        P_climb_total       = self.mission.P_climb_shaft
        P_cs_total, T_v2, V_2, gamma_min = self._cs25_121_power()

        if P_cs_total >= P_climb_total:
            P_total = P_cs_total
            driver  = "CS-25.121 second segment"
        else:
            P_total = P_climb_total
            driver  = "Climb (mission peak)"

        P_per_eng         = P_total / cfg.N_engines
        T_static_per_eng  = self._static_thrust(P_per_eng)
        T_static_total    = cfg.N_engines * T_static_per_eng

        return PowerSizingBreakdown(
            P_TO_total          = P_total,
            P_TO_per_engine     = P_per_eng,
            P_from_climb        = P_climb_total,
            P_from_CS25_121     = P_cs_total,
            driving_case        = driver,
            T_per_engine_V2     = T_v2,
            V_2                 = V_2,
            LD_takeoff          = cfg.LD_takeoff,
            gamma_min           = gamma_min,
            T_static_total      = T_static_total,
            T_static_per_engine = T_static_per_eng,
        )