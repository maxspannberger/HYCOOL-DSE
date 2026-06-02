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
    climb_eff: float = 1.0
    cruise_eff: float = 1.0
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
    Wing_taper: float = 0.0
    l_f_m: float = 0.0

    def summary(self):
        status_color = "green" if self.converged else "red"
        main_info = (
            f"MTOW: {self.MTOW/1000:.2f} t\n"
            f"OEW:  {(self.W_empty+cfg.W_fixed)/1000:.2f} t\n"
            f"Fuel: {self.W_fuel:.1f} kg\n"
            f"Payload: {self.W_payload/1000:.1f} t\n"
            f"Iterations: {self.iterations} \n"
            f"Wing Area: {self.Wing_Area:.2f} m^2\n"
            f"Wing Span: {self.Wing_span:.2f} m\n"
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


def compute_wing_geometry(MTOW: float, cfg: AircraftConfig,M_cruise:float) -> tuple[float, float, float, float, float]:
    """
    Trapezoidal wing planform at constant wing loading, AR.

    Returns (S_ref, b, c_root, c_tip, MAC) and wing taper changes in the config file.
    """
    from dataclasses import replace as _replace
    if M_cruise >=0.66:
        sweep=np.arccos(1.16/(M_cruise+0.5))
    else:
        sweep=0.0
    
    lam = 0.2*(2-sweep)
    S_ref = MTOW * G / cfg.Loading
    b = np.sqrt(cfg.AR * S_ref)
    c_root = 2 * S_ref / ((1 + lam) * b)
    c_tip = lam * c_root
    MAC = (2 / 3) * c_root * (1 + lam + lam**2) / (1 + lam)


    return S_ref, b, c_root, c_tip, MAC,lam


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
    S_ref, b, c_root, c_tip, MAC,taper = compute_wing_geometry(cfg.MTOW_initial, cfg, cfg.M_cruise)
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
        config = 3

    if config == 1:
        hump_tank = True
    else:
        hump_tank = False

    for it in range(1, max_iter + 1):

        # Recompute wing planform at the start of each iteration: under
        # constant wing loading, AR, and taper, S_ref, b, c_root, c_tip,
        # MAC all scale with the current MTOW.
        S_ref, b, c_root, c_tip, MAC,taper = compute_wing_geometry(MTOW, cfg, cfg.M_cruise)

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
            OEW_kg       = wt_bd.W_empty+cfg.W_fixed,
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
                f"OEW={wt_bd.W_empty+cfg.W_fixed:7.1f} kg)")

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
    S_ref, b, c_root, c_tip, MAC,taper = compute_wing_geometry(MTOW, cfg, cfg.M_cruise)
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
        climb_eff = wt_bd.climb_eff,
        cruise_eff = wt_bd.cruise_eff,
        P_TO_KW     = P_TO_kW,
        P_TO_OEI_KW = P_TO_OEI_kW,
        P_cruise_KW = P_cruise_kw,
        P_max_KW    = P_max_kw,
        P_climb_KW  = P_climb_kW,
        P_reserve_KW= P_reserve_kw,
        t_climb=t_climb,
        t_cruise=t_cruise,
        Wing_Area=S_ref,
        Wing_span=b,
        Wing_taper=taper,
        l_f_m=cfg_iter.l_f
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
    while M_cruise>=0.6:
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
            "Wing_area": cfg_updated.S_ref,
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

    return best_row

def compute_additional_aerodynamic_parameters(best_row: dict | None, cfg_updated: AircraftConfig) -> dict:

    if best_row is None:
        best_row = get_optimal_cl_mach(cfg_updated)

    from dataclasses import replace
    cfg_updated = replace(cfg_updated, M_cruise=best_row['M_cruise'])

    # Adjust CL for compressibility effects using Prandtl-Glauert correction
    beta=1-best_row['M_cruise']**2

    CL_adjusted = best_row['CL_cruise'] / np.sqrt(beta)

        # get CLalpha for the current Mach number and aspect ratio with datcom equation
    airfoilefficiency=0.95
    CLalpha=2*np.pi*cfg_updated.AR/(2+np.sqrt(4+(cfg_updated.AR*np.sqrt(beta)/airfoilefficiency)**2*(1+np.tan(best_row['half_sweep'])**2)/(beta**2)))

    #find trim angle to fly at CL_cruise with the adjusted CLalpha, define lift angle of attack as alpha0 according to the chosen airfoil

    alpha0=-2.5 * np.pi/180 # for example, for a typical airfoil, TODO: replace with actual alpha0 for the chosen airfoil

    alpha_trim=CL_adjusted/CLalpha+alpha0

    #calculate Reynolds number at cruise conditions
    mu_alt=1.158e-5       # dynamic viscosity of air at 7620 m in kg/(m*s)
    T,p,rho=isa(cfg_updated.altitude_cruise)
    a_cruise=np.sqrt(1.4*287.05*T) # speed of sound at cruise altitude
    v_cruise=best_row['M_cruise']*a_cruise
    Reynolds=rho*v_cruise*cfg_updated.MAC/mu_alt

    #Calculate the maximum Lift coefficient for landing configuration
    rho_ground=1.225        # sea level standard density in kg/m^3
    V_stall=best_row['Stall Speed']
    M_landing=best_row['MTOW']-(best_row['m_LH2_cruise']+best_row['m_LH2_climb']+best_row['m_LH2_taxi_TO'])
    CL_max_LD=2*(M_landing)*G/(rho_ground*best_row['Wing Area']*V_stall**2)

    #Calculate the increase in CL_max for takeoff and landing due to high-lift devices (flaps, slats). These are rough estimates and can be refined with more detailed aerodynamic analysis or empirical data.
    c_fowler_c_wing=0.2 # assume fowler flaps make up 20% wing chord
    x_c_hinge=1-c_fowler_c_wing

    #calculate the hinge sweep angle
    hinge_sweep=np.arctan(np.tan(best_row['half_sweep'])-x_c_hinge*2*best_row['root_chord']/(best_row['span'])*(1-best_row['taper']))

    #accoridng to NASA paper, a deflection angle of 30 degrees was most effective for the fowler flap, so we get deltac/cf
    deltac_cf=0.55      #extracted from the figure in toreenbeek

    #get the fraction of the increase of the chord with extended fowler flaps with respect to original chord
    deltac_c=deltac_cf+c_fowler_c_wing
    cdash_c=1+deltac_c

    #get increase in Clmax for landing and takeoff according to flap use, takeoff lower deflection wanted
    deltaClmax_LD=1.3*cdash_c
    deltaClmax_TO=deltaClmax_LD*0.6

    #get increase in Clmax for landing and takeoff according to LE HLD use, takeoff lower deflection wanted
    le_flap_area_wing_ratio = 0.6          #assume 60% of wing area used for slats
    deltaClmax_LE_LD=0.3
    deltaCLmax_LE_LD=0.9*deltaClmax_LE_LD*le_flap_area_wing_ratio*np.cos(best_row['LE_sweep'])
    deltaCLmax_LE_TO=deltaCLmax_LE_LD*0.6

    #get wing deltaCLmax increases for takeoff and landing
    deltaCL_max_TO=best_row['CL_max_TO']-best_row['CL_max_clean']-deltaCLmax_LE_TO
    deltaCL_max_LD=CL_max_LD-best_row['CL_max_clean']-deltaCLmax_LE_LD

    #calculate the area needed for flaps to achieve the desired increase in CLmax
    flap_area_TO_ratio=(deltaCL_max_TO)/(deltaClmax_TO)*1/(0.9*np.cos(hinge_sweep))
    flap_area_LD_ratio=(deltaCL_max_LD)/(deltaClmax_LD)*1/(0.9*np.cos(hinge_sweep))

    if flap_area_TO_ratio >= flap_area_LD_ratio:
        te_flap_area_wing = flap_area_TO_ratio
        driving= "takeoff"
    else:
        te_flap_area_wing = flap_area_LD_ratio
        driving="landing"

    best_row["CL_prandtl"] = CL_adjusted
    best_row["CLalpha"] = CLalpha
    best_row["alpha_trim"] = alpha_trim
    best_row["CL_max_LD"] = CL_max_LD
    best_row["delta_Cl_max_TO"] = deltaClmax_TO
    best_row["delta_Cl_max_LD"] = deltaClmax_LD
    best_row["LE_flap_area_wing"] = le_flap_area_wing_ratio
    best_row["TE_flap_area_wing"] = te_flap_area_wing
    best_row["hinge_sweep_deg"] = hinge_sweep * 180 / np.pi

    print(best_row["Aileron_Area_ratio"])

    print()
    print("[bold]Optimal Flight Conditions:[/bold]")
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("value", justify="right")
    table.add_column("M_cruise", justify="right")
    table.add_column("CL_cruise", justify="right")
    table.add_column("CL_prandtl", justify="right")
    table.add_column("CLalpha [1/degree]", justify="right")
    table.add_column("alpha_trim [degree]", justify="right")
    table.add_column("CD_total", justify="right")
    table.add_column("Fuel savings due to adjusted Mach [kg]", justify="right")
    table.add_row(
        f"{best_row['value']:.6f}",
        f"{best_row['M_cruise']:.2f}",
        f"{best_row['CL_cruise']:.6f}",
        f"{best_row['CL_prandtl']:.6f}",
        f"{best_row['CLalpha']*np.pi/180:.6f}",
        f"{best_row['alpha_trim']*180/np.pi:.2f}",
        f"{best_row['CD_total']:.6f}",
        f"{best_row['fuel_savings']:.2f}",
    )
    print(table)

    print("[bold]Wing Aerodynamics:[/bold]")
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("CL_max_TO", justify="right")
    table.add_column("CL_max_LD", justify="right")
    table.add_column("delta CL_max_TO", justify="right")
    table.add_column("delta CL_max_LD", justify="right")
    table.add_column("LE Area [m^2]", justify="right")
    table.add_column("TE Area [m^2]", justify="right")
    table.add_column("Flap Hinge Sweep [degree]", justify="right")
    table.add_column("Driving Condition", justify="right")
    table.add_column("Reynolds Number", justify="right")
    table.add_row(
        f"{best_row['CL_max_TO']:.6f}",
        f"{best_row['CL_max_LD']:.6f}",
        f"{deltaCL_max_TO:.6f}",
        f"{deltaCL_max_LD:.6f}",
        f"{best_row['LE_flap_area_wing']:.2f}",
        f"{best_row['TE_flap_area_wing']:.2f}",
        f"{best_row['hinge_sweep_deg']:.2f}",
        f"  {driving}",
        f"{Reynolds:.0f}"
    )
    print(table)
    return dict(
        CL_opt=best_row['CL_cruise'],
        CD_Dmin=best_row['CD_total'],
        M_opt=best_row['M_cruise'],
        value=best_row['value'],
        t_cruise=best_row['t_cruise'],
        m_LH2_cruise=best_row['m_LH2_cruise'],
        fuel_savings=best_row['fuel_savings'],
        W_landing=M_landing,
        CL_max_LD=best_row["CL_max_LD"],
        delta_Cl_flap=deltaCL_max_LD,
        CL_alpha0_flapped=cfg_updated.CL_alpha0_clean+deltaCL_max_LD,
        cdash_c=cdash_c,
        TE_flap_area_wing=te_flap_area_wing,
        taper=best_row['taper'],
        MAC=cfg_updated.MAC,
        
    )

if __name__ == "__main__":
    cfg = default_q400_hycool()
    # result1 = run_class_ii(cfg,comp=comp_params, tol=1.0, max_iter=100, verbose=True)

    # print()
    # print(result1.drag.summary())
    # print()
    # print(result1.weight.summary())
    # print()
    # print(result1.mission.summary())
    # print()
    # print(result1.summary())

    # paths = export_results(
    #     result1,
    #     output_dir = "outputs",
    #     iterations = result1.iteration_log,
    # )
    # print()
    # print("[bold]Results exported to:[/bold]")
    # for label, p in paths.items():
    #     print(f"  {label}: {p}")

    best_row = get_optimal_cl_mach(cfg, force_recompute=True)
    cfg = apply_optimal_cl_mach(cfg, best_row)
    compute_additional_aerodynamic_parameters(best_row, cfg)