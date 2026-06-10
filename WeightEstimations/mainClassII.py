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
from dataclasses import dataclass, replace
import sys
from pathlib import Path
import json

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from WeightEstimations.Aircraft_Config   import AircraftConfig, default_q400_hycool
from WeightEstimations.ClassII_Tail       import TailSizing_Input,  TailSizingEstimator, TailSizingBreakdown
from WeightEstimations.ClassII_Drag   import ClassII_Drag_Input, DragEstimation,      DragBreakdown
from WeightEstimations.ClassII_Weight import ClassII_Input,      weightEstimation,    WeightBreakdown
from WeightEstimations.Mission_Power     import MissionPower,       MissionFuelBreakdown
from WeightEstimations.Power_Sizing      import PowerSizing,        PowerSizingBreakdown
from WeightEstimations.Export_Results    import export_results
from General.component_parameters import component_params as comp_params
from WeightEstimations.ISA import isa
from WeightEstimations.Export_Geometry import (
    print_final_geometry,
    export_final_geometry,
)


from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table

G = 9.80665

_optimal_cl_mach_cache: dict | None = None
_optimal_cl_mach_path = Path(__file__).resolve().parent / "outputs" / "optimal_cl_mach_cache.json"


def apply_optimal_cl_mach(cfg: AircraftConfig, best_row: dict) -> AircraftConfig:
    """Return a copy of cfg with the optimal cruise Mach applied."""
    from dataclasses import replace
    return replace(cfg, M_cruise=best_row["M_cruise"])


def get_optimal_cl_mach(cfg: AircraftConfig, force_recompute: bool = False) -> dict:
    """Return the cached optimal Mach result, computing it only once unless forced."""
    global _optimal_cl_mach_cache
    if _optimal_cl_mach_cache is None and not force_recompute:
        if _optimal_cl_mach_path.exists():
            with open(_optimal_cl_mach_path, "r", encoding="utf-8") as f:
                _optimal_cl_mach_cache = json.load(f)

    if _optimal_cl_mach_cache is None or force_recompute:
        _optimal_cl_mach_cache = find_optimal_cl_mach(cfg, force_recompute=force_recompute)
    return _optimal_cl_mach_cache.copy()


@dataclass
class ClassIIResult:

    MTOW:        float
    MZFW:        float
    W_empty:     float
    OEW:         float
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
    aeroparameters: dict
    tail_rechecked: TailSizingBreakdown    # rerun with computed T_TO
    iteration_log: list = None             # per-iteration MTOW trace
    total_prop_efficiency: float = 1.0
    climb_eff: float = 1.0
    cruise_eff: float = 1.0
    bt_charging_ratio: float = 0.0

    P_TO_KW: float = 0.0
    P_TO_OEI_KW: float = 0.0
    P_cruise_KW: float = 0.0
    P_max_KW: float = 0.0
    P_climb_KW: float = 0.0
    P_reserve_KW: float = 0.0
    P_approach_KW: float = 0.0

    t_climb: float = 0.0
    t_cruise: float = 0.0

    Wing_Area: float = 0.0
    Wing_span: float = 0.0
    Wing_taper: float = 0.0
    l_f_m: float = 0.0
    root_chord: float = 0.0
    MAC: float = 0.0
    l_f: float = 0.0
    Wing_sweep_quarter: float = 0.0
    Wing_sweep_half: float = 0.0
    Wing_sweep_LE: float = 0.0
    distance_le_mac_to_cg: float = 0.0
    distance_le_mac_to_turbine: float = 0.0
    distance_le_root_to_le_mac: float = 0.0

    def summary(self):
        status_color = "green" if self.converged else "red"
        main_info = (
            f"MTOW: {self.MTOW/1000:.2f} t\n"
            f"OEW:  {(self.OEW)/1000:.2f} t\n"
            f"Fuel: {self.W_fuel:.1f} kg\n"
            f"Payload: {self.W_payload/1000:.1f} t\n"
            f"Iterations: {self.iterations} \n"
            f"Wing Area: {self.Wing_Area:.2f} m^2\n"
            f"Wing Span: {self.Wing_span:.2f} m\n"
            f"Root Chord: {self.root_chord:.2f} m\n"
            f"Tip Chord: {self.root_chord*self.Wing_taper:.2f} m\n"
            f"MAC: {self.MAC:.2f} m\n"
            f"Wing Sweep (Quarter): {self.Wing_sweep_quarter*180/np.pi:.2f} deg\n"
            f"Wing Sweep (Half): {self.Wing_sweep_half*180/np.pi:.2f} deg\n"
            f"Wing Sweep (LE): {self.Wing_sweep_LE*180/np.pi:.2f} deg\n"
            f"Fuselage Diameter: {self.l_f:.2f} m\n"
            f"Distance from MAC Leading Edge to CG: {self.distance_le_mac_to_cg:.2f} m\n"
            f"Distance from MAC Leading Edge to Turbine: {self.distance_le_mac_to_turbine:.2f} m\n"
            f"Distance from Root Chord Leading Edge to MAC Leading Edge: {self.distance_le_root_to_le_mac:.2f} m\n"
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
    #r_tank = (3.0 * V_tank / (10.0 * np.pi)) ** (1 / 3) if V_tank > 0 else 0.0
    # L_tank = 4.0 * r_tank
    # d_tank = 2.0 * r_tank
    # r_tank = (3.0 * V_tank / (4.0 * np.pi)) ** (1 / 3) if V_tank > 0 else 0.0
    # d_tank = 2.0 * r_tank
    # L_tank = d_tank

    r_tank = (cfg.diameter_margin * cfg.b_f_i - 2*cfg.wall_thickness)/2
    d_tank = 2*r_tank
    l_cyl_tank = (V_tank-(4/3)*np.pi*(r_tank**3))/(np.pi*(r_tank**2))
    L_tank = 2*r_tank + l_cyl_tank

    # ---------- calculation for cylindrical tank with hemispherical endcaps --------------

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


def compute_wing_geometry(MTOW: float, cfg: AircraftConfig,M_cruise:float,c_root: float) -> tuple[float, float, float, float, float]:
    """
    Trapezoidal wing planform at constant wing loading, AR.

    Returns (S_ref, b, c_root, c_tip, MAC) and wing taper changes in the config file.
    """
    from dataclasses import replace as _replace
    if M_cruise >=0.66:
        sweep_quarter=np.arccos(1.16/(M_cruise+0.5))
    else:
        sweep_quarter=0.0

    lam = 0.2*(2-sweep_quarter)
    S_ref = MTOW * G / cfg.Loading
    b = np.sqrt(cfg.AR * S_ref)

    c_root = 2 * S_ref / ((1 + lam) * b)

    sweep_LE=np.arctan(sweep_quarter+0.25*2*c_root/b*(1-lam))

    sweep_half=np.arctan(sweep_LE-0.5*c_root/b*(1-lam))
    
    c_tip = lam * c_root
    MAC = (2 / 3) * c_root * (1 + lam + lam**2) / (1 + lam)

    cfg_updated = _replace(cfg, S_ref=S_ref, b=b, c_root=c_root, MAC=MAC,taper=lam,sweep_half=sweep_half,sweep_tc=sweep_quarter)

    return  cfg_updated, S_ref, b, c_root, c_tip, MAC,lam, sweep_quarter, sweep_half, sweep_LE


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
    cfg_updated, S_ref, b, c_root, c_tip, MAC,taper, sweep_quarter, sweep_half, sweep_LE = compute_wing_geometry(cfg.MTOW_initial, cfg, cfg.M_cruise, c_root=cfg.c_root)
    y_engine_4 = cfg_updated.b / 2 * (2 / 3)
    cfg_updated = replace(cfg_updated, y_engine_4=y_engine_4)
    tail_inp = TailSizing_Input.from_config(cfg_updated, S_ref=S_ref, b=b, MAC=MAC)
    tail_bd  = TailSizingEstimator(tail_inp).compute()

    if verbose:
        print(tail_bd.summary())
        print()

    # -----------------------------------------------------------------
    # Step 2: outer MTOW iteration with mission-power coupling
    # -----------------------------------------------------------------
    MTOW    = cfg.MTOW_initial
    W_fixed = cfg.W_fixed_frn * MTOW
    W_fuel  = cfg.m_dot_fuel * cfg.range_m / cfg.V_cruise
    drag_bd = DragBreakdown()
    wt_bd   = WeightBreakdown()
    mis_bd  = MissionFuelBreakdown()
    converged = False
    bt_charging_ratio = 0.0
    it = 0
    iteration_log: list[dict] = []
    CL_approach=1.53            #initial guess for the approach CL

    if config is None:
        config = 3

    if config == 1:
        hump_tank = True
    else:
        hump_tank = False

    for it in range(1, max_iter + 1):

        # Recompute wing planform at the start of each iteration: under
        # constant wing loading, AR, and taper, S_ref, b, c_root, c_tip,
        # MAC all scale with the current MTOW.
        cfg_updated, S_ref, b, c_root, c_tip, MAC,taper, sweep_quarter, sweep_half, sweep_LE = compute_wing_geometry(MTOW, cfg, cfg.M_cruise, c_root=c_root)
        y_engine_4 = cfg_updated.b / 2 * (2 / 3)
        cfg_updated = replace(cfg_updated, y_engine_4=y_engine_4)

        # Recompute fuselage / H2-tank geometry from the latest W_fuel.
        cfg_iter, r_tank, L_tank, d_tank, S_wet_hump = compute_fuselage_geometry(W_fuel, cfg_updated,hump_back=hump_tank)

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
            Sweep_quarter = sweep_quarter,
            Sweep_half = sweep_half,
        )
        drag_bd = DragEstimation(drag_inp).compute()

        # Mission power -> LH2 fuel mass
        mis_bd = MissionPower(cfg_iter, drag_bd, config=config,comp=comp, MTOW=MTOW, S_ref=S_ref,CL_approach=CL_approach).compute()
        M_landing = MTOW - (
            mis_bd.m_LH2_cruise
            + mis_bd.m_LH2_climb
            + mis_bd.m_LH2_TO_taxi
        )        
        W_fuel = mis_bd.m_LH2_total
        P_max_kw = mis_bd.P_max / 1000
        P_cruise_kw = mis_bd.P_cruise_shaft / 1000
        P_reserve_kw = mis_bd.P_reserve_shaft / 1000
        P_approach_kW = mis_bd.P_app_shaft / 1000
        t_cruise = mis_bd.t_cruise
        t_climb = mis_bd.t_climb
        t_reserve = mis_bd.t_reserve

        # Performance & CS-25 Requirements
        pwr_bd = PowerSizing(cfg_iter, mis_bd, MTOW).compute()
        P_TO_kW = pwr_bd.P_TO_total / 1000.0
        P_TO_OEI_kW = pwr_bd.P_total_OEI / 1000.0
        # P_climb_kW = pwr_bd.P_from_climb / 1000.0
        P_climb_kW = mis_bd.P_climb_shaft / 1000

        cfg_tail = replace_T_TO(cfg_iter, pwr_bd.T_static_per_engine)

        tail_inp = TailSizing_Input.from_config(
            cfg_tail,
            MTOW=MTOW,
            S_ref=S_ref,
            b=b,
            MAC=MAC,
            M_landing = M_landing,
        )

        tail_bd = TailSizingEstimator(tail_inp).compute()

        aero_parameters=compute_additional_aerodynamic_parameters(cfg_iter, drag_bd, mis_bd, pwr_bd,sweep_half,MAC,MTOW,\
                                                                  S_ref,b,taper,c_root,sweep_quarter,sweep_LE,verbose=False)
        #M_landing = aero_parameters["W_landing"]

        # Weight at current MTOW with sized tails and current wing geometry
        wt_inp = ClassII_Input.from_config(
            cfg_iter,
            aero_parameters=aero_parameters,
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
            max_Thrust_prop_inner = pwr_bd.T_static_per_prop_inner,
            max_Thrust_prop_outer = pwr_bd.T_static_per_prop_outer,
            W_fuel = W_fuel,
            configuration=config,
            t_climb=t_climb,
            t_cruise=t_cruise,
            t_reserve=t_reserve,
            base_params=False,
            bt_charging_ratio = bt_charging_ratio,
            taper = taper,
            sweep_LE = sweep_LE
        )
        # print("WEIGHT INPUT TAIL:")
        # print(f"S_v used in weight = {wt_inp.S_v:.3f} m²")
        # print(f"b_v used in weight = {wt_inp.b_v:.3f} m")
        # print(f"tail_bd S_v        = {tail_bd.S_v:.3f} m²")

        wt_bd = weightEstimation(wt_inp, comp).compute()

        # Close the loop
        MZFW_new = wt_bd.W_empty + cfg.W_payload + W_fixed
        MTOW_new = MZFW_new + W_fuel
        bt_charging_ratio = wt_bd.bt_charging_ratio
        delta    = abs(MTOW_new - MTOW)
        OEW_kg       = wt_bd.W_empty+W_fixed

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
            OEW_kg       = OEW_kg,
            bt_ch_ratio  = bt_charging_ratio
        ))

        if verbose:
            print(f"  iter {it:2d}: MTOW {MTOW:8.1f} -> {MTOW_new:8.1f} kg  "
                f"(S={S_ref:5.2f} m^2, b={b:5.2f} m, "
                f"c_r={c_root:4.2f} m, MAC={MAC:4.2f} m, "
                f"l_f={cfg_iter.l_f:5.2f} m, d_tank={d_tank:4.2f} m, "
                f"S_v={tail_bd.S_v:5.2f} m^2, "
                f"b_v={tail_bd.b_v:4.2f} m, "
                f"W_vtail={wt_bd.W_vtail:6.1f} kg, "
                f"L/D={drag_bd.L_over_D:5.2f}, "
                f"P_cr={mis_bd.P_cruise_shaft/1000:5.0f} kW, "
                f"fuel={W_fuel:6.1f} kg, "
                f"OEW={wt_bd.W_empty+W_fixed:7.1f} kg)")

        MTOW = MTOW_new

        aero_parameters=compute_additional_aerodynamic_parameters(cfg_iter, drag_bd, mis_bd, pwr_bd,sweep_half,MAC,MTOW,\
                                                                  S_ref,b,taper,c_root,sweep_quarter,sweep_LE,verbose=False)

        CL_approach = aero_parameters["CL_approach"]
        if delta < tol:
            converged = True
            break

    MZFW = wt_bd.W_empty + cfg.W_payload + W_fixed

    # -----------------------------------------------------------------
    # Step 3 (post-loop): power & takeoff thrust sizing
    # -----------------------------------------------------------------
    pwr_bd = PowerSizing(cfg_iter, mis_bd, MTOW).compute()

    if verbose:
        print()
        print(pwr_bd.summary())


    aero_parameters=compute_additional_aerodynamic_parameters(cfg_iter, drag_bd, mis_bd, pwr_bd,sweep_half,MAC,MTOW,\
                                                                  S_ref,b,taper,c_root,sweep_quarter,sweep_LE,verbose=True)
    
    M_landing = aero_parameters["W_landing"]

    # -----------------------------------------------------------------
    # Step 4 (post-loop): re-run tail sizing with computed T_TO
    # so the OEI rudder check uses a self-consistent thrust value
    # rather than the user-supplied initial guess.
    # -----------------------------------------------------------------
    cfg_recheck = replace_T_TO(cfg_iter, pwr_bd.T_static_per_engine)
    y_engine_4 = cfg_recheck.b / 2 * (2 / 3)
    cfg_recheck = replace(cfg_recheck, y_engine_4=y_engine_4)
    cfg_updated, S_ref, b, c_root, c_tip, MAC,taper, sweep_quarter, sweep_half, sweep_LE = compute_wing_geometry(MTOW, cfg_recheck, cfg_recheck.M_cruise, c_root=c_root)
    y_engine_4 = cfg_updated.b / 2 * (2 / 3)
    cfg_updated = replace(cfg_updated, y_engine_4=y_engine_4)
    tail_inp_recheck = TailSizing_Input.from_config(
        cfg_updated, MTOW=MTOW, S_ref=S_ref, b=b, MAC=MAC, M_landing = M_landing,
    )
    tail_bd_recheck  = TailSizingEstimator(tail_inp_recheck).compute()

    # print("RECHECKED TAIL:")
    # print(f"tail_rechecked S_v = {tail_bd_recheck.S_v:.3f} m²")
    # print(f"tail_rechecked b_v = {tail_bd_recheck.b_v:.3f} m")

    if verbose:
        print()
        print("Tail sizing rechecked with computed T_TO:")
        print(tail_bd_recheck.summary())

    print(pwr_bd.gamma_min_prop)
    print(pwr_bd.gamma_min_engine)

    cgwingpos = b / 2 * 0.35
    turbinewingpos = cfg_updated.b_f / 2 *0.5       #inner engine position chosen at half of half span

    # c(y) = c_root * [1 - (1 - lambda) * 2y/b] for a trapezoidal wing
    taper_slope = 1.0 - taper
    chordatcgpos = c_root * (1.0 - taper_slope * (cgwingpos / (b / 2)))
    turbinechord=c_root*(1-taper_slope*(turbinewingpos/(b/2)))

    macchorddiff = c_root - MAC
    machspanpos = (
        macchorddiff / (c_root * taper_slope) * (b / 2)
        if abs(taper_slope) > 1e-9
        else 0.0
    )
    print(f"MAC chord: {MAC:.3f} m, corresponding spanwise position: {machspanpos:.3f} m")
    print(f"CG chord position: {chordatcgpos:.3f} m, corresponding spanwise position: {cgwingpos:.3f} m")
    print(f"Turbine chord position: {turbinechord:.3f} m, corresponding spanwise position: {turbinewingpos:.3f} m")



    cgalong_chord = (0.7 * chordatcgpos - 0.15 * chordatcgpos) * 0.7+0.15*chordatcgpos

    # print(cfg_updated.d_f/2-machspanpos) 

    #distance from Root chord leading edge to MAC leading edge
    distance_le_root_to_le_mac = np.tan(sweep_LE)*(cfg_updated.d_f/2-machspanpos)


    # Distance from MAC leading edge (front edge) CG location
    distance_le_mac_to_cg = np.tan(sweep_LE)*(cgwingpos-machspanpos)+cgalong_chord

    #distance from the LE of the inside propeller to the LEMAC
    distance_le_mac_to_turbine = np.tan(sweep_LE)*(turbinewingpos-machspanpos)

    return ClassIIResult(
        MTOW       = MTOW,
        MZFW       = MZFW,
        W_empty    = wt_bd.W_empty,
        OEW        = OEW_kg,
        W_fuel     = mis_bd.m_LH2_total,
        W_payload  = cfg.W_payload,
        W_fixed    = W_fixed,
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
        climb_eff = wt_bd.climb_eff,
        cruise_eff = wt_bd.cruise_eff,
        P_TO_KW     = P_TO_kW,
        P_TO_OEI_KW = P_TO_OEI_kW,
        P_cruise_KW = P_cruise_kw,
        P_max_KW    = P_max_kw,
        P_climb_KW  = P_climb_kW,
        P_reserve_KW= P_reserve_kw,
        P_approach_KW = P_approach_kW,
        t_climb=t_climb,
        t_cruise=t_cruise,
        Wing_Area=S_ref,
        Wing_span=b,
        Wing_taper=taper,
        l_f_m=cfg_iter.l_f,
        root_chord=c_root,
        MAC=MAC,
        l_f=cfg_updated.d_f,
        Wing_sweep_quarter=sweep_quarter,
        Wing_sweep_half=sweep_half,
        Wing_sweep_LE=sweep_LE,
        aeroparameters=aero_parameters,
        distance_le_mac_to_cg=distance_le_mac_to_cg,
        distance_le_mac_to_turbine=distance_le_mac_to_turbine,
        distance_le_root_to_le_mac=distance_le_root_to_le_mac
    )


def replace_T_TO(cfg: AircraftConfig, T_TO_new: float) -> AircraftConfig:
    """Return a copy of cfg with T_TO_per_engine updated."""
    from dataclasses import replace
    return replace(cfg, T_TO_per_engine=T_TO_new)

def find_optimal_cl_mach(cfg: AircraftConfig, force_recompute: bool = False) -> dict:
    # This function is a placeholder for the actual optimal CL and Mach calculation.
    # In a real implementation, this would involve more complex logic and possibly
    # iterative methods to find the optimal values based on the aircraft configuration.
    global _optimal_cl_mach_cache
    if _optimal_cl_mach_cache is not None and not force_recompute:
        return _optimal_cl_mach_cache.copy()

    from dataclasses import replace
    M_cruise=0.7
    sweep_rows=[]
    iterations=0
    while M_cruise>=0.59:
        factor=0.01
        cfg_updated = replace(cfg, M_cruise=M_cruise)
        result = run_class_ii(cfg_updated,comp=comp_params, tol=1.0, max_iter=100, verbose=False)
        value = cfg_updated.M_cruise*result.drag.CL_cruise/result.drag.CD_total
        CL_cruise = result.drag.CL_cruise
        
        sweep_rows.append({
            "value": value,
            "M_cruise": cfg_updated.M_cruise,
            "CL_cruise": CL_cruise,
            "CD_total": result.drag.CD_total,
            "t_cruise": result.mission.t_cruise,
            "m_LH2_cruise": result.mission.m_LH2_cruise,
            "m_LH2_climb": result.mission.m_LH2_climb,
            "m_LH2_taxi_TO": result.mission.m_LH2_TO_taxi,
            "MTOW": result.MTOW,
            "Wing Area": result.Wing_Area,
            "Stall Speed": cfg_updated.V_stall,
            "CL_max_TO": result.power.CL_max_TO,
            "CL_max_clean": cfg_updated.CL_max,
            "half_sweep": cfg_updated.sweep_half,
            "LE_sweep": cfg_updated.sweep_tc,
            "root_chord": cfg_updated.c_root,
            "span": cfg_updated.b,
            "taper": result.Wing_taper,
            "Aileron_Area_ratio": result.tail_rechecked.Sa_Sref,
        })

        iterations+=1
        print(f"Completed iteration {iterations} with M_cruise={M_cruise:.2f}, value={value:.6f}, CL_cruise={CL_cruise:.6f}, CD_total={result.drag.CD_total:.6f}, t_cruise={result.mission.t_cruise/60:.2f} min, m_LH2_cruise={result.mission.m_LH2_cruise:.2f} kg")
        M_cruise=M_cruise-factor

    reference_row = max(sweep_rows, key=lambda row: row["m_LH2_cruise"])
    reference_fuel = reference_row["m_LH2_cruise"]
    for row in sweep_rows:
        row["fuel_savings"] = reference_fuel - row["m_LH2_cruise"]

    best_row = max(sweep_rows, key=lambda row: row["value"])
    _optimal_cl_mach_cache = best_row.copy()

    _optimal_cl_mach_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_optimal_cl_mach_path, "w", encoding="utf-8") as f:
        json.dump(_optimal_cl_mach_cache, f, indent=4)

    print(best_row)

    return best_row

def compute_additional_aerodynamic_parameters(cfg_updated: AircraftConfig,drag_result: dict,mission: dict,power: dict,\
                                              sweep_half: float,MAC: float,MTOW: float,Wing_Area: float,\
                                               Wing_span: float,Wing_taper: float,root_chord: float,\
                                                Wing_sweep_quarter: float,Wing_sweep_LE: float,verbose: bool ) -> dict:
    aero = {}
    # if best_row is None:
    #     best_row = get_optimal_cl_mach(cfg_updated)

    # from dataclasses import replace
    # cfg_updated = replace(cfg_updated, M_cruise=0.68)

    # result = run_class_ii(cfg_updated,comp=comp_params, tol=1.0, max_iter=100, verbose=False)


    # Adjust CL for compressibility effects using Prandtl-Glauert correction
    beta=1-cfg_updated.M_cruise**2

    CL_adjusted = drag_result.CL_cruise / np.sqrt(beta)

        # get CLalpha for the current Mach number and aspect ratio with datcom equation
    airfoilefficiency=0.95
    CLalpha=2*np.pi*cfg_updated.AR/(2+np.sqrt(4+(cfg_updated.AR*np.sqrt(beta)/airfoilefficiency)**2*(1+np.tan(sweep_half)**2)/(beta**2)))

    #find trim angle to fly at CL_cruise with the adjusted CLalpha, define lift angle of attack as alpha0 according to the chosen airfoil

    alphaCL0=cfg_updated.alpha_CL0_clean 

    alpha_trim=CL_adjusted/CLalpha+alphaCL0

    #calculate Reynolds number at cruise conditions
    mu_alt=1.158e-5       # dynamic viscosity of air at 7620 m in kg/(m*s)
    T,p,rho=isa(cfg_updated.altitude_cruise)
    a_cruise=np.sqrt(1.4*287.05*T) # speed of sound at cruise altitude
    v_cruise=cfg_updated.M_cruise*a_cruise
    Reynolds_cruise=rho*v_cruise*MAC/mu_alt

    #calculate Reynolds number at sea level for takeoff and landing
    mu_ground=1.7894e-5   # dynamic viscosity of air at sea level in kg/(m*s)
    a_ground=np.sqrt(1.4*287.05*288.15) # speed of sound at sea level
    v_stall=cfg_updated.V_stall
    Reynolds_ground=rho*v_stall*MAC/mu_ground

    #Calculate the maximum Lift coefficient for landing configuration
    rho_ground=1.225        # sea level standard density in kg/m^3
    V_stall=cfg_updated.V_stall
    M_landing=MTOW-(mission.m_LH2_cruise+mission.m_LH2_climb+mission.m_LH2_TO_taxi)
    CL_max_LD=2*(M_landing)*G/(rho_ground*Wing_Area*V_stall**2)
    CL_max_LD_without_stall=2*M_landing*G/(rho_ground*Wing_Area*(V_stall*1.3)**2)

    #Calculate the increase in CL_max for takeoff and landing due to high-lift devices (flaps, slats). These are rough estimates and can be refined with more detailed aerodynamic analysis or empirical data.
    c_fowler_c_wing=0.3 # assume fowler flaps make up 30% wing chord
    x_c_hinge=1-c_fowler_c_wing

    #calculate the hinge sweep angle
    hinge_sweep=np.arctan(np.tan(Wing_sweep_LE)-x_c_hinge*2*root_chord/(Wing_span)*(1-Wing_taper))

    #accoridng to NASA paper, a deflection angle of 30 degrees was most effective for the fowler flap, so we get deltac/cf
    deltac_cf=0.55      #extracted from the figure in toreenbeek

    #get the fraction of the increase of the chord with extended fowler flaps with respect to original chord
    deltac_c=deltac_cf*c_fowler_c_wing
    cdash_c=1+deltac_c

    Clneeded=CL_adjusted/(np.cos(Wing_sweep_quarter)**2)

    #get increase in Clmax for landing and takeoff according to flap use, takeoff lower deflection wanted
    deltaClmax_LD=1.3*cdash_c
    deltaClmax_TO=deltaClmax_LD*0.6

    #get increase in Clmax for landing and takeoff according to LE HLD use, takeoff lower deflection wanted
    le_flap_area_wing_ratio = 0.7          #assume 70% of wing area used for slats
    deltaClmax_LE_LD=0.3                #assume slats give 0.3 increase in Clmax for landing
    deltaCLmax_LE_LD=0.9*deltaClmax_LE_LD*le_flap_area_wing_ratio*np.cos(Wing_sweep_LE)
    deltaCLmax_LE_TO=deltaCLmax_LE_LD*0.6

    #get wing deltaCLmax for flaps used for increases for takeoff and landing
    deltaCL_max_TO=power.CL_max_TO-cfg_updated.CL_max-deltaCLmax_LE_TO
    deltaCL_max_LD=CL_max_LD-cfg_updated.CL_max-deltaCLmax_LE_LD

    #calculate the area needed for flaps to achieve the desired increase in CLmax
    flap_area_TO_ratio=(deltaCL_max_TO)/(deltaClmax_TO)*1/(0.9*np.cos(hinge_sweep))
    flap_area_LD_ratio=(deltaCL_max_LD)/(deltaClmax_LD)*1/(0.9*np.cos(hinge_sweep))

    if flap_area_TO_ratio >= flap_area_LD_ratio:
        te_flap_area_wing = flap_area_TO_ratio
        driving= "takeoff"
        deltaCL_max_LD_new=deltaCL_max_LD*te_flap_area_wing/flap_area_LD_ratio
        aero["delta_CL_max_LD"] = deltaCL_max_LD_new
    else:
        te_flap_area_wing = flap_area_LD_ratio
        driving="landing"
        deltaCL_max_TO_new=deltaCL_max_TO*te_flap_area_wing/flap_area_TO_ratio
        aero["CL_max_TO with new area"] = deltaCL_max_TO_new

    aero["CL_prandtl"] = CL_adjusted
    aero["CLalpha"] = CLalpha
    aero["alpha_trim"] = alpha_trim
    aero["CL_cruise"] = drag_result.CL_cruise
    aero["CD_total"] = drag_result.CD_total
    aero["CL_max_TO"] = power.CL_max_TO
    aero["CL_max_LD"] = CL_max_LD
    aero["CL_approach"] = CL_max_LD_without_stall
    aero["delta_Cl_max_TO"] = deltaClmax_TO
    aero["delta_Cl_max_LD"] = deltaClmax_LD
    aero["LE_flap_area_wing"] = le_flap_area_wing_ratio
    aero["TE_flap_area_wing"] = te_flap_area_wing
    aero["hinge_sweep_deg"] = hinge_sweep * 180 / np.pi

    value=cfg_updated.M_cruise*drag_result.CL_cruise/drag_result.CD_total

    if verbose==True:
        
        print()
        print("[bold]Optimal Flight Conditions:[/bold]")
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("value", justify="right")
        table.add_column("M_cruise", justify="right")
        table.add_column("CL_cruise", justify="right")
        table.add_column("CL_prandtl", justify="right")
        table.add_column("Cl_needed_airfoil", justify="right")
        table.add_column("CLalpha [1/degree]", justify="right")
        table.add_column("alpha_trim [degree]", justify="right")
        table.add_column("CD_total", justify="right")
        table.add_row(
            f"{value:.6f}",
            f"{cfg_updated.M_cruise:.2f}",
            f"{aero['CL_cruise']:.6f}",
            f"{aero['CL_prandtl']:.6f}",
            f"{Clneeded:.6f}",
            f"{aero['CLalpha']*np.pi/180:.6f}",
            f"{aero['alpha_trim']*180/np.pi:.2f}",
            f"{aero['CD_total']:.6f}",
        )
        print(table)

        print("[bold]Wing Aerodynamics:[/bold]")
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("CL_max_TO", justify="right")
        table.add_column("CL_max_LD", justify="right")
        table.add_column("approach speed landing CL", justify="right")
        table.add_column("delta CL_max_TO", justify="right")
        table.add_column("delta CL_max_LD", justify="right")
        table.add_column("LE Area [m^2]", justify="right")
        table.add_column("TE Area [m^2]", justify="right")
        table.add_column("Flap Hinge Sweep [degree]", justify="right")
        table.add_column("Driving Condition", justify="right")
        table.add_column("Reynolds Number at cruise", justify="right")
        table.add_column("Reynolds Number at ground", justify="right")
        table.add_row(
            f"{aero['CL_max_TO']:.6f}",
            f"{aero['CL_max_LD']:.6f}",
            f"{CL_max_LD_without_stall:.6f}",
            f"{deltaCL_max_TO:.6f}",
            f"{deltaCL_max_LD:.6f}",
            f"{aero['LE_flap_area_wing']:.2f}",
            f"{aero['TE_flap_area_wing']:.2f}",
            f"{aero['hinge_sweep_deg']:.2f}",
            f"  {driving}",
            f"{Reynolds_cruise:.0f}",
            f"{Reynolds_ground:.0f}"
        )
        print(table)


    return dict(
        CL_opt=aero['CL_cruise'],
        CD_Dmin=aero['CD_total'],
        M_opt=cfg_updated.M_cruise,
        value=value,
        m_LH2_cruise=mission.m_LH2_cruise,
        W_landing=M_landing,
        CL_max_LD=aero['CL_max_LD'],
        CL_max_LD_approach=CL_max_LD_without_stall,
        delta_CL_flap=deltaCL_max_LD,
        delta_Cl_flap=deltaClmax_LD,
        CL_alpha0_flapped=cfg_updated.CL_alpha0_clean+deltaCL_max_LD,
        CL_max_TO_with_new_area=aero['CL_max_TO with new area'],
        cdash_c=cdash_c,
        TE_flap_area_wing=te_flap_area_wing,
        delta_defl=30*np.pi/180,
        hinge_sweep=hinge_sweep,
        b_fs=te_flap_area_wing*Wing_span,
        S_f=te_flap_area_wing*Wing_Area,
        S_LE=le_flap_area_wing_ratio*Wing_Area,
        taper=Wing_taper,
        MAC=MAC,
        CL_approach=aero['CL_approach']

        
    )

if __name__ == "__main__":
    cfg = default_q400_hycool()
    result1 = run_class_ii(cfg,comp=comp_params, tol=1.0, max_iter=100, verbose=True)

    print_final_geometry(cfg, result1)

    paths = export_final_geometry(
        cfg,
        result1,
        output_dir="outputs",
    )

    print("\nFinal geometry exported to:")
    for label, path in paths.items():
        print(f"{label}: {path}")

    print()
    print(result1.drag.summary())
    print()
    print(result1.weight.summary())
    print()
    print(result1.mission.summary())
    print()
    print(result1.summary())

    paths = export_results(
        result1,
        output_dir = "outputs",
        iterations = result1.iteration_log,
    )
    print()
    print("[bold]Results exported to:[/bold]")
    for label, p in paths.items():
        print(f"  {label}: {p}")

    #set price of LH2 per kg
    cost_per_kg_LH2 = 3.0       #€/kg, which is an estimate for 2050

    cfg_2prop = replace(cfg, N_propellers=2)
    result2 = run_class_ii(cfg_2prop, comp=comp_params, tol=1.0, max_iter=100, verbose=False)
    
    fuelsavings = result2.W_fuel - result1.W_fuel
    costsavings = fuelsavings * cost_per_kg_LH2

    # MTOW change due to using 4 props (result1) vs 2 props (result2)
    mtow_diff_kg = result1.MTOW - result2.MTOW
    mtow_pct = (mtow_diff_kg / result2.MTOW * 100.0) if result2.MTOW != 0 else 0.0
    # signed display and human-readable word
    mtow_word = "increase" if mtow_diff_kg > 0 else ("decrease" if mtow_diff_kg < 0 else "no change")

    # Propulsion system mass change (4-prop - 2-prop)
    propmass_diff_kg = result1.W_prop - result2.W_prop
    propmass_pct = (propmass_diff_kg / result2.W_prop * 100.0) if result2.W_prop != 0 else 0.0

    # Highlight fuel, cost savings and MTOW impact in a panel to make them stand out
    savings_text = (
        f"[bold white]Switching to 4 propellers saves[/bold white]\n"
        f"[bold green]{fuelsavings:.1f} kg[/bold green]\n"
        f"[bold white]Estimated cost savings per flight:[/bold white] [bold yellow]€{costsavings:.2f}[/bold yellow]\n"
        f"[bold white]MTOW change (4 prop - 2 prop):[/bold white] [bold]{mtow_diff_kg:+.1f} kg[/bold] "
        f"[bold white]({mtow_word}, {mtow_pct:+.2f}% vs 2-prop)[/bold white]\n"
        f"[bold white]Propulsion mass change (4 prop - 2 prop):[/bold white] [bold]{propmass_diff_kg:+.1f} kg[/bold] "
        f"[bold white]({propmass_pct:+.2f}% vs 2-prop)[/bold white]"
    )
    print(Panel(savings_text, title="[bold cyan]Fuel, Cost & MTOW Impact[/bold cyan]", border_style="cyan", expand=False))

    #get_optimal_cl_mach(cfg, force_recompute=True)

    print(result1.P_approach_KW)
