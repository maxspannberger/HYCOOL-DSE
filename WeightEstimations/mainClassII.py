"""
main.py

Class II loop tying together tail sizing, drag, weight estimation, and
mission power / fuel calculation.

Structure:

    1. Run tail sizing once, outside the loop. (No MTOW dependence.)

    2. Iterate on MTOW:
        a. Compute drag at current MTOW with sized tails.
        b. Run MissionPower with current drag breakdown:
              - Cruise   : level flight at h_cruise, M_cruise
              - Climb    : (D*V + W*ROC)/eta_prop at midpoint altitude
              - Reserve  : 45 min hold at 1500 ft, 1.3 V_stall
              - Takeoff/taxi: 2% allowance of (cruise + climb) fuel
              - Takeoff power : reference output, not added to fuel
           Each phase uses eta_prop and eta_thermal to convert
           shaft power to LH2 mass flow via LHV = 120 MJ/kg.
        c. Compute structural weight at current MTOW with sized tails.
        d. New MTOW = OEW + payload + total LH2 fuel + fixed.
        e. Check |MTOW_new - MTOW_old| < tol.

    Power requirements per phase are exposed in result.mission so the
    propulsion group can be sized against P_max.
"""

import numpy as np
from dataclasses import dataclass

from Aircraft_Config   import AircraftConfig, default_q400_hycool
from ClassII_Tail       import TailSizing_Input,  TailSizingEstimator, TailSizingBreakdown
from ClassII_Drag   import ClassII_Drag_Input, DragEstimation,      DragBreakdown
from ClassII_Weight import ClassII_Input,      weightEstimation,    WeightBreakdown
from Mission_Power     import MissionPower,       MissionFuelBreakdown
from Power_Sizing      import PowerSizing,        PowerSizingBreakdown


G = 9.80665


@dataclass
class ClassIIResult:

    MTOW:        float
    MZFW:        float
    W_empty:     float
    W_fuel:      float
    W_payload:   float
    W_fixed:     float
    L_over_D:    float
    CL_cruise:   float
    iterations:  int
    converged:   bool

    tail:        TailSizingBreakdown
    drag:        DragBreakdown
    weight:      WeightBreakdown
    mission:     MissionFuelBreakdown
    power:       PowerSizingBreakdown
    tail_rechecked: TailSizingBreakdown    # rerun with computed T_TO

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        lines = [
            "#" * 60,
            "  Class II Integrated Sizing Result  --  " + status,
            "#" * 60,
            f"  Iterations         {self.iterations}",
            f"  MTOW               {self.MTOW:>10.1f} kg",
            f"  MZFW               {self.MZFW:>10.1f} kg",
            f"  OEW (W_empty)      {self.W_empty:>10.1f} kg",
            f"  Payload            {self.W_payload:>10.1f} kg",
            f"  Fuel (LH2)         {self.W_fuel:>10.1f} kg",
            f"  Fixed equipment    {self.W_fixed:>10.1f} kg",
            f"  CL cruise          {self.CL_cruise:>10.4f}",
            f"  L/D cruise         {self.L_over_D:>10.2f}",
            f"  P_cruise (shaft)   {self.mission.P_cruise_shaft/1000:>10.1f} kW",
            f"  P_climb  (shaft)   {self.mission.P_climb_shaft/1000:>10.1f} kW",
            f"  P_max    (shaft)   {self.mission.P_max/1000:>10.1f} kW",
            f"  P_TO required      {self.power.P_TO_total/1000:>10.1f} kW  ({self.power.driving_case})",
            f"  T_static per eng   {self.power.T_static_per_engine/1000:>10.2f} kN",
            "#" * 60,
        ]
        return "\n".join(lines)


def run_class_ii(
    cfg:      AircraftConfig,
    tol:      float = 1.0,
    max_iter: int   = 100,
    verbose:  bool  = True,
) -> ClassIIResult:

    # -----------------------------------------------------------------
    # Step 1: tail sizing (does not depend on MTOW)
    # -----------------------------------------------------------------
    tail_inp = TailSizing_Input.from_config(cfg)
    tail_bd  = TailSizingEstimator(tail_inp).compute()

    if verbose:
        print(tail_bd.summary())
        print()

    # -----------------------------------------------------------------
    # Step 2: outer MTOW iteration with mission-power coupling
    # -----------------------------------------------------------------
    MTOW    = cfg.MTOW_initial
    drag_bd = DragBreakdown()
    wt_bd   = WeightBreakdown()
    mis_bd  = MissionFuelBreakdown()
    converged = False
    it = 0

    for it in range(1, max_iter + 1):

        # Drag at current MTOW with sized tails
        drag_inp = ClassII_Drag_Input.from_config(
            cfg,
            MTOW = MTOW,
            S_h  = tail_bd.S_h,
            S_v  = tail_bd.S_v,
        )
        drag_bd = DragEstimation(drag_inp).compute()

        # Mission power -> LH2 fuel mass
        mis_bd = MissionPower(cfg, drag_bd, MTOW).compute()
        W_fuel = mis_bd.m_LH2_total

        # Weight at current MTOW with sized tails
        wt_inp = ClassII_Input.from_config(
            cfg,
            MTOW = MTOW,
            MZFW = MTOW - W_fuel,
            S_h  = tail_bd.S_h,
            S_v  = tail_bd.S_v,
            b_v  = tail_bd.b_v,
        )
        wt_bd = weightEstimation(wt_inp).compute()

        # Close the loop
        MZFW_new = wt_bd.W_empty + cfg.W_payload + cfg.W_fixed
        MTOW_new = MZFW_new + W_fuel
        delta    = abs(MTOW_new - MTOW)

        if verbose:
            print(f"  iter {it:2d}: MTOW {MTOW:8.1f} -> {MTOW_new:8.1f} kg  "
                  f"(L/D={drag_bd.L_over_D:5.2f}, "
                  f"P_cr={mis_bd.P_cruise_shaft/1000:5.0f} kW, "
                  f"fuel={W_fuel:6.1f} kg, "
                  f"OEW={wt_bd.W_empty:7.1f} kg)")

        MTOW = MTOW_new

        if delta < tol:
            converged = True
            break

    MZFW = wt_bd.W_empty + cfg.W_payload + cfg.W_fixed

    # -----------------------------------------------------------------
    # Step 3 (post-loop): power & takeoff thrust sizing
    # -----------------------------------------------------------------
    pwr_bd = PowerSizing(cfg, mis_bd, MTOW).compute()

    if verbose:
        print()
        print(pwr_bd.summary())

    # -----------------------------------------------------------------
    # Step 4 (post-loop): re-run tail sizing with computed T_TO
    # so the OEI rudder check uses a self-consistent thrust value
    # rather than the user-supplied initial guess.
    # -----------------------------------------------------------------
    cfg_recheck = replace_T_TO(cfg, pwr_bd.T_static_per_engine)
    tail_inp_recheck = TailSizing_Input.from_config(cfg_recheck, MTOW=MTOW)
    tail_bd_recheck  = TailSizingEstimator(tail_inp_recheck).compute()

    if verbose:
        print()
        print("Tail sizing rechecked with computed T_TO:")
        print(tail_bd_recheck.summary())

    return ClassIIResult(
        MTOW       = MTOW,
        MZFW       = MZFW,
        W_empty    = wt_bd.W_empty,
        W_fuel     = mis_bd.m_LH2_total,
        W_payload  = cfg.W_payload,
        W_fixed    = cfg.W_fixed,
        L_over_D   = drag_bd.L_over_D,
        CL_cruise  = drag_bd.CL_cruise,
        iterations = it,
        converged  = converged,
        tail       = tail_bd,
        drag       = drag_bd,
        weight     = wt_bd,
        mission    = mis_bd,
        power      = pwr_bd,
        tail_rechecked = tail_bd_recheck,
    )


def replace_T_TO(cfg: AircraftConfig, T_TO_new: float) -> AircraftConfig:
    """Return a copy of cfg with T_TO_per_engine updated."""
    from dataclasses import replace
    return replace(cfg, T_TO_per_engine=T_TO_new)


if __name__ == "__main__":
    cfg = default_q400_hycool()
    result = run_class_ii(cfg, tol=1.0, max_iter=100, verbose=True)

    print()
    print(result.drag.summary())
    print()
    print(result.weight.summary())
    print()
    print(result.mission.summary())
    print()
    print(result.summary())