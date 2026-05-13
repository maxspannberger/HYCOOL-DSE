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
import sys
from pathlib import Path

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from Aircraft_Config   import AircraftConfig, default_q400_hycool
from ClassII_Tail       import TailSizing_Input,  TailSizingEstimator, TailSizingBreakdown
from ClassII_Drag   import ClassII_Drag_Input, DragEstimation,      DragBreakdown
from ClassII_Weight import ClassII_Input,      weightEstimation,    WeightBreakdown
from Mission_Power     import MissionPower,       MissionFuelBreakdown
from Power_Sizing      import PowerSizing,        PowerSizingBreakdown
from Export_Results    import export_results
from General.component_parameters import component_params as comp_params


from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

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
    iteration_log: list = None             # per-iteration MTOW trace

    def summary(self):
        status_color = "green" if self.converged else "red"
        main_info = (
            f"MTOW: {self.MTOW/1000:.2f} t\n"
            f"OEW:  {self.W_empty/1000:.2f} t\n"
            f"Fuel: {self.W_fuel:.1f} kg\n"
            f"Payload: {self.W_payload/1000:.1f} t\n"
            f"Iterations: {self.iterations}"
        )
        
        perf_info = (
            f"Cruise L/D: [bold]{self.L_over_D:.2f}[/bold]\n"
            f"Climb Shaft Power: {self.mission.P_max/1000000:.2f} MW\n"
            f"Max Shaft Power: {self.power.P_from_CS25_121/1000000:.2f} MW\n"
            f"Static Thrust/Eng: {self.power.T_static_per_engine/1000:.2f} kN"
        )

        return Panel(
            Columns([main_info, perf_info]),
            title=f"[bold {status_color}]Class II Integrated Sizing Result[/bold {status_color}]",
            border_style=status_color
        )


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
    iteration_log: list[dict] = []
    config = int(input("Enter config for power unit weight estimation (1-4): "))
    comp=comp_params

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
        P_max_kw = mis_bd.P_max / 1000
        P_cruise_kw = mis_bd.P_cruise_shaft / 1000

        # Performance & CS-25 Requirements
        pwr_bd = PowerSizing(cfg, mis_bd, MTOW).compute()
        P_TO_kW = pwr_bd.P_TO_total / 1000.0
        P_TO_OEI_kW = pwr_bd.P_total_OEI / 1000.0


        # Weight at current MTOW with sized tails
        wt_inp = ClassII_Input.from_config(
            cfg,
            comp=comp,
            MTOW = MTOW,
            MZFW = MTOW - W_fuel,
            S_h  = tail_bd.S_h,
            S_v  = tail_bd.S_v,
            b_v  = tail_bd.b_v,
            P_TO_KW = P_TO_kW,
            P_TO_OEI_KW = P_TO_OEI_kW,
            P_cruise_KW=P_cruise_kw,
            P_max_KW = P_max_kw,
            W_fuel = W_fuel,
            configuration=config
        )
        wt_bd = weightEstimation(wt_inp).compute()

        # Close the loop
        MZFW_new = wt_bd.W_empty + cfg.W_payload + cfg.W_fixed
        MTOW_new = MZFW_new + W_fuel
        delta    = abs(MTOW_new - MTOW)

        iteration_log.append(dict(
            iter         = it,
            MTOW_in_kg   = MTOW,
            MTOW_out_kg  = MTOW_new,
            delta_kg     = delta,
            L_over_D     = drag_bd.L_over_D,
            P_cruise_kW  = mis_bd.P_cruise_shaft / 1000,
            P_max_kW     = mis_bd.P_max / 1000,
            P_TO_kW      = P_TO_kW,
            W_fuel_kg    = W_fuel,
            OEW_kg       = wt_bd.W_empty,
        ))

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
        iteration_log  = iteration_log,
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

    paths = export_results(
        result,
        output_dir = "outputs",
        iterations = result.iteration_log,
    )
    print()
    print("[bold]Results exported to:[/bold]")
    for label, p in paths.items():
        print(f"  {label}: {p}")