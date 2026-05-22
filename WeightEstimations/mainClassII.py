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
    W_prop:      float
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
    total_prop_efficiency: float = 1.0
    bt_charging_ratio: float = 0.0

    P_TO_KW: float = 0.0
    P_TO_OEI_KW: float = 0.0
    P_cruise_KW: float = 0.0
    P_max_KW: float = 0.0
    P_climb_KW: float = 0.0
    P_reserve_KW: float = 0.0

    t_climb: float = 0.0
    t_cruise: float = 0.0

    Wing_Area: float = 0.0
    Wing_span: float = 0.0

    def summary(self):
        status_color = "green" if self.converged else "red"
        main_info = (
            f"MTOW: {self.MTOW/1000:.2f} t\n"
            f"OEW:  {self.W_empty/1000:.2f} t\n"
            f"Fuel: {self.W_fuel:.1f} kg\n"
            f"Payload: {self.W_payload/1000:.1f} t\n"
            f"Iterations: {self.iterations} \n"
            f"Wing Area: {self.Wing_Area:.2f} m^2\n"
            f"Wing Span: {self.Wing_span:.2f} m"
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


def compute_fuselage_geometry(W_fuel: float, cfg: AircraftConfig,hump_back: bool):
    """
    Resize the fuselage to accommodate the LH2 tank.

    Tank: cylindrical core of length 2r with two hemispherical end caps,
    giving V_tank = (10/3) pi r^3. Total tank length L_tank = 4r and
    tank diameter d_tank = 2r.

    Hump-back layout (747-style): the tank rides on top of the fuselage.
    Fuselage length and width are unchanged; the exposed half of the
    tank's surface (approx. 4 pi r^2) is added to the fuselage wetted
    area.

    In-line layout: the tank lies inside the fuselage, so the fuselage
    grows by L_tank in length. If the tank diameter exceeds the
    pre-defined fuselage diameter, b_f and h_f are increased to the
    tank diameter. The fuselage wetted area is re-evaluated with the
    same teardrop formula used in the weight module so the two stay
    consistent.

    Returns (cfg_updated, r_tank, L_tank, d_tank, S_wet_hump).
    """
    from dataclasses import replace as _replace

    V_tank = max(W_fuel, 0.0) / cfg.rho_LH2_eff
    r_tank = (3.0 * V_tank / (10.0 * np.pi)) ** (1 / 3) if V_tank > 0 else 0.0
    L_tank = 4.0 * r_tank
    d_tank = 2.0 * r_tank

    if hump_back:
        l_f = cfg.l_f
        b_f = cfg.b_f
        h_f = cfg.h_f
        S_wet_hump = 4.0 * np.pi * r_tank**2
        S_wet_f = cfg.S_wet_f + S_wet_hump
    else:
        S_wet_hump = 0.0
        l_f = cfg.l_f + L_tank
        b_f = max(cfg.b_f, d_tank)
        h_f = max(cfg.h_f, d_tank)
        d_eq = 0.5 * (b_f + h_f)
        sigma = l_f / d_eq
        S_wet_f = (
            np.pi * b_f * l_f
            * (1.0 - 2.0 / sigma) ** (2.0 / 3.0)
            * (1.0 + 1.0 / sigma**2)
        )

    cfg_updated = _replace(cfg, l_f=l_f, b_f=b_f, h_f=h_f, S_wet_f=S_wet_f)
    return cfg_updated, r_tank, L_tank, d_tank, S_wet_hump


def compute_wing_geometry(MTOW: float, cfg: AircraftConfig) -> tuple[float, float, float, float, float]:
    """
    Trapezoidal wing planform at constant wing loading, AR, and taper.

    Returns (S_ref, b, c_root, c_tip, MAC).
    """
    lam = cfg.taper
    S_ref = MTOW * G / cfg.Loading
    b = np.sqrt(cfg.AR * S_ref)
    c_root = 2 * S_ref / ((1 + lam) * b)
    c_tip = lam * c_root
    MAC = (2 / 3) * c_root * (1 + lam + lam**2) / (1 + lam)
    return S_ref, b, c_root, c_tip, MAC


def run_class_ii(
    cfg:        AircraftConfig,
    comp:       dict,
    tol:        float   = 1.0,
    max_iter:   int     = 100,
    verbose:    bool    = True,
    config:     int     = None, 
) -> ClassIIResult:

    # -----------------------------------------------------------------
    # Step 1: tail sizing (uses initial wing geometry; refined in the recheck)
    # -----------------------------------------------------------------
    S_ref, b, c_root, c_tip, MAC = compute_wing_geometry(cfg.MTOW_initial, cfg)
    tail_inp = TailSizing_Input.from_config(cfg, S_ref=S_ref, b=b, MAC=MAC)
    tail_bd  = TailSizingEstimator(tail_inp).compute()

    if verbose:
        print(tail_bd.summary())
        print()

    # -----------------------------------------------------------------
    # Step 2: outer MTOW iteration with mission-power coupling
    # -----------------------------------------------------------------
    MTOW    = cfg.MTOW_initial
    W_fuel  = cfg.m_dot_fuel * cfg.range_m / cfg.V_cruise
    drag_bd = DragBreakdown()
    wt_bd   = WeightBreakdown()
    mis_bd  = MissionFuelBreakdown()
    converged = False
    bt_charging_ratio = 0.0
    it = 0
    iteration_log: list[dict] = []

    if config is None:
        config = int(input("Enter config for power unit weight estimation (1-5): "))

    if config == 1:
        hump_tank = True
    else:
        hump_tank = False

    for it in range(1, max_iter + 1):

        # Recompute wing planform at the start of each iteration: under
        # constant wing loading, AR, and taper, S_ref, b, c_root, c_tip,
        # MAC all scale with the current MTOW.
        S_ref, b, c_root, c_tip, MAC = compute_wing_geometry(MTOW, cfg)

        # Recompute fuselage / H2-tank geometry from the latest W_fuel.
        cfg_iter, r_tank, L_tank, d_tank, S_wet_hump = compute_fuselage_geometry(W_fuel, cfg,hump_back=hump_tank)

        # Drag at current MTOW with sized tails and current wing geometry
        drag_inp = ClassII_Drag_Input.from_config(
            cfg_iter,
            MTOW = MTOW,
            S_ref = S_ref,
            b = b,
            c_root = c_root,
            MAC = MAC,
            S_h  = tail_bd.S_h,
            S_v  = tail_bd.S_v,
        )
        drag_bd = DragEstimation(drag_inp).compute()

        # Mission power -> LH2 fuel mass
        mis_bd = MissionPower(cfg_iter, drag_bd, config=config, MTOW=MTOW, S_ref=S_ref).compute()
        W_fuel = mis_bd.m_LH2_total
        P_max_kw = mis_bd.P_max / 1000
        P_cruise_kw = mis_bd.P_cruise_shaft / 1000
        P_reserve_kw = mis_bd.P_reserve_shaft / 1000
        t_cruise = mis_bd.t_cruise
        t_climb = mis_bd.t_climb
        t_reserve = mis_bd.t_reserve

        # Performance & CS-25 Requirements
        pwr_bd = PowerSizing(cfg, mis_bd, MTOW).compute()
        P_TO_kW = pwr_bd.P_TO_total / 1000.0
        P_TO_OEI_kW = pwr_bd.P_total_OEI / 1000.0
        P_climb_kW = pwr_bd.P_from_climb / 1000.0



        # Weight at current MTOW with sized tails and current wing geometry
        wt_inp = ClassII_Input.from_config(
            cfg_iter,
            comp = comp,
            MTOW = MTOW,
            MZFW = MTOW - W_fuel,
            S_ref = S_ref,
            b = b,
            S_h  = tail_bd.S_h,
            S_v  = tail_bd.S_v,
            b_v  = tail_bd.b_v,
            P_TO_KW = P_TO_kW,
            P_TO_OEI_KW = P_TO_OEI_kW,
            P_cruise_KW=P_cruise_kw,
            P_max_KW = P_max_kw,
            P_climb_KW=P_climb_kW,
            P_reserve_KW=P_reserve_kw,
            W_fuel = W_fuel,
            configuration=config,
            t_climb=t_climb,
            t_cruise=t_cruise,
            t_reserve=t_reserve,
            base_params=False,
            bt_charging_ratio = bt_charging_ratio,
        )
        wt_bd = weightEstimation(wt_inp, comp).compute()

        # Close the loop
        MZFW_new = wt_bd.W_empty + cfg.W_payload + cfg.W_fixed
        MTOW_new = MZFW_new + W_fuel
        bt_charging_ratio = wt_bd.bt_charging_ratio
        delta    = abs(MTOW_new - MTOW)

        iteration_log.append(dict(
            iter         = it,
            MTOW_in_kg   = MTOW,
            MTOW_out_kg  = MTOW_new,
            delta_kg     = delta,
            S_ref_m2     = S_ref,
            b_m          = b,
            c_root_m     = c_root,
            c_tip_m      = c_tip,
            MAC_m        = MAC,
            r_tank_m     = r_tank,
            L_tank_m     = L_tank,
            d_tank_m     = d_tank,
            l_f_m        = cfg_iter.l_f,
            d_f_m        = cfg_iter.d_f,
            S_wet_f_m2   = cfg_iter.S_wet_f,
            S_wet_hump_m2= S_wet_hump,
            L_over_D     = drag_bd.L_over_D,
            P_cruise_kW  = mis_bd.P_cruise_shaft / 1000,
            P_max_kW     = mis_bd.P_max / 1000,
            P_TO_kW      = P_TO_kW,
            W_fuel_kg    = W_fuel,
            OEW_kg       = wt_bd.W_empty,
            bt_ch_ratio  = bt_charging_ratio
        ))

        if verbose:
            print(f"  iter {it:2d}: MTOW {MTOW:8.1f} -> {MTOW_new:8.1f} kg  "
                f"(S={S_ref:5.2f} m^2, b={b:5.2f} m, "
                f"c_r={c_root:4.2f} m, MAC={MAC:4.2f} m, "
                f"l_f={cfg_iter.l_f:5.2f} m, d_tank={d_tank:4.2f} m, "
                f"L/D={drag_bd.L_over_D:5.2f}, "
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
    pwr_bd = PowerSizing(cfg_iter, mis_bd, MTOW).compute()

    if verbose:
        print()
        print(pwr_bd.summary())

    # -----------------------------------------------------------------
    # Step 4 (post-loop): re-run tail sizing with computed T_TO
    # so the OEI rudder check uses a self-consistent thrust value
    # rather than the user-supplied initial guess.
    # -----------------------------------------------------------------
    cfg_recheck = replace_T_TO(cfg_iter, pwr_bd.T_static_per_engine)
    S_ref, b, c_root, c_tip, MAC = compute_wing_geometry(MTOW, cfg)
    tail_inp_recheck = TailSizing_Input.from_config(
        cfg_recheck, MTOW=MTOW, S_ref=S_ref, b=b, MAC=MAC,
    )
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
        W_prop     = wt_bd.W_total_prop,
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
        total_prop_efficiency = wt_bd.total_prop_efficiency,
        P_TO_KW     = P_TO_kW,
        P_TO_OEI_KW = P_TO_OEI_kW,
        P_cruise_KW = P_cruise_kw,
        P_max_KW    = P_max_kw,
        P_climb_KW  = P_climb_kW,
        P_reserve_KW= P_reserve_kw,
        t_climb=t_climb,
        t_cruise=t_cruise,
        Wing_Area=S_ref,
        Wing_span=b
    )


def replace_T_TO(cfg: AircraftConfig, T_TO_new: float) -> AircraftConfig:
    """Return a copy of cfg with T_TO_per_engine updated."""
    from dataclasses import replace
    return replace(cfg, T_TO_per_engine=T_TO_new)


if __name__ == "__main__":
    cfg = default_q400_hycool()
    result = run_class_ii(cfg,comp=comp_params, tol=1.0, max_iter=100, verbose=True)

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