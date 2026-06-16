import numpy as np
from numpy.random import normal as normal
from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import WeightEstimations.Tail_Interpolation as Tail_Interp
from WeightEstimations.Aircraft_Config import AircraftConfig
from General.component_parameters import component_params as comp_params
from Propulsion.efficiency import GT_BAT_efficiency, GT_FC_efficiency, GT_GT_efficiency,FC_BAT_efficiency
from Propulsion.electrical import perform_complete_electrical_sizing
from Propulsion.main import run as run_gt_sizing
from H2_piping.main import main_H2_nominal
from H2_piping.main_OEI import main_H2_OEI
from H2_piping.weight_calcs import pipe_calculations
from Storage.mainStorage import main_storage

from rich.table import Table



@dataclass
class ClassII_Input:
    # Weights
    MTOW:       float = 0.0
    MZFW:       float = 0.0
    W_fixed:    float = 0.0
    n_ult:      float = 3.75

    # Wing
    b:          float = 0.0
    S_w:        float = 0.0
    sweep_half: float = 0.0
    sweep_LE:   float = 0.0
    taper:      float = 0.0
    t_r:        float = 0.0
    k_w:        float = 6.67e-3
    const_wing: float = 4.58e-3
    b_ref:     float = 1.905
    k_e:       float = 0.0
    k_uc:      float = 1.0
    k_st:     float = 0.0
    const_k_st: float = 9.06e-4
    k_b:      float = 1.0
    const_HLD: float = 2.706
    k_f1:      float = 1.15
    k_f2:      float = 1.0
    W_hld_LE: float = 20.0              #kg/m^2 according to figure on torenbeek page 454

    # Horizontal tail
    S_h:        float = 0.0
    sweep_h:    float = 0.0

    # Vertical tail
    S_v:        float = 0.0
    sweep_v:    float = 0.0
    b_v:        float = 0.0
    t_tail:     bool  = False
    h_h:        float = 0.0
    k_v:        float = 0.0

    # Fuselage
    b_f:        float = 0.0
    h_f:        float = 0.0
    l_f:        float = 0.0
    l_t:        float = 0.0
    k_wf:       float = 0.23

    # Speed
    V_dive:     float = 0.0
    V_stall:    float = 0.0
    t_c_flap:   float = 0.0

    # Landing gear
    high_wing:  bool = False

    # Surface controls
    has_flap_slat: bool = True

    # Propulsion
    rho_turb:    float = 0.0
    rho_bat:     float = 0.0
    rho_fc:      float = 0.0
    rho_HTS_gen: float = 0.0
    rho_HTS_pow: float = 0.0
    rho_ac_dc:    float = 0.0
    rho_dc_dc_1:    float = 0.0
    rho_dc_dc_2:    float = 0.0
    rho_dc_ac:    float = 0.0
    rho_cable:    float = 0.0
    rho_cable:    float = 0.0
    rho_pipe:     float = 0.0
    P_TO_KW:     float = 0.0
    P_max_KW:    float = 0.0
    P_cruise_KW: float = 0.0
    P_TO_OEI_KW: float = 0.0
    P_climb_KW:  float = 0.0
    P_reserve_KW: float = 0.0
    P_takeoff_KW: float = 0.0
    P_approach_KW: float = 0.0
    max_Thrust_prop_inner: float = 0.0
    max_Thrust_prop_outer: float = 0.0
    W_fuel:      float = 0.0
    grav_density:float = 0.64
    configuration: int = 1
    cable_lentgh: float = 0.0
    pipe_length:  float = 0.0
    N_engines: float = 0.0
    N_propellers: int = 1

    #flight phase times
    t_cruise: float = 0.0
    t_climb: float = 0.0
    t_reserve: float = 0.0
    bt_charging_ratio: float = 0.0
    P_opt: float = 0.0
    mass_margin: float = 1.05
    aero_parameters: dict = field(default_factory=dict)

    #TMS masses
    TMS_mass_N2: float = 0,
    TMS_mass_N4: float = 0,

    frn_tank_support: float = 0,

    H2_results_all: dict = None

    @classmethod
    def from_config(
        cls,
        cfg: AircraftConfig,
        comp: dict,
        aero_parameters: Optional[dict] = None,
        MTOW: Optional[float] = None,
        MZFW: Optional[float] = None,
        W_fixed: Optional[float] = None,
        S_ref: Optional[float] = None,
        b:    Optional[float] = None,
        S_h:  Optional[float] = None,
        S_v:  Optional[float] = None,
        b_v:  Optional[float] = None,
        P_TO_KW:  float = 0.0,
        P_max_KW: float = 0.0,
        W_fuel:   float = 0.0,
        configuration: int = 1,
        P_cruise_KW: float = 0.0,
        P_climb_KW: float = 0.0,
        P_reserve_KW: float = 0.0,
        P_TO_OEI_KW: float = 0.0,
        P_takeoff_KW: float = 0.0,
        P_approach_KW: float = 0.0,
        max_Thrust_prop_inner: float = 0.0,
        max_Thrust_prop_outer: float = 0.0,
        t_cruise: float = 0.0,
        t_climb: float = 0.0,
        t_reserve: float = 0.0,
        N_engines: float = 0.0,
        N_propellers: int = 1,
        base_params: bool = False,
        bt_charging_ratio: float = 0.0,
        P_opt: float = 0.0,
        mass_margin: float = 1.1,
        taper: float = 0.0,
        sweep_LE: float = 0.0,
        H2_results_all: dict = None
    ) -> "ClassII_Input":
        """
        Build the weight-estimator input from a shared AircraftConfig.

        S_h, S_v, b_v default to the initial guesses in cfg but should be
        overridden by the tail-sizing module's outputs each iteration.
    S_ref defaults to cfg.S_ref but should be overridden by the
    iteratively updated wing area (S_ref = MTOW*g/Loading).
        MTOW defaults to cfg.MTOW_initial.
        """
        return cls(
            MTOW          = MTOW if MTOW is not None else cfg.MTOW_initial,
            MZFW          = MZFW if MZFW is not None else cfg.MTOW_initial * 0.95,
            W_fixed       = W_fixed, 
            n_ult         = cfg.n_ult,

            b             = b     if b     is not None else cfg.b,
            S_w           = S_ref if S_ref is not None else cfg.S_ref,
            sweep_half    = cfg.sweep_half,
            taper=taper,
            sweep_LE=sweep_LE,
            t_r           = cfg.t_root_abs,

            S_h           = S_h if S_h is not None else cfg.S_h_initial,
            sweep_h       = cfg.sweep_h_half,

            S_v           = S_v if S_v is not None else cfg.S_v_initial,
            sweep_v       = cfg.sweep_v_half,
            b_v           = b_v if b_v is not None else cfg.b_v_initial,
            t_tail        = cfg.t_tail,
            h_h           = cfg.h_h,

            b_f           = cfg.b_f,
            h_f           = cfg.h_f,
            l_f           = cfg.l_f,
            l_t           = cfg.l_t,

            V_dive        = cfg.V_dive,
            V_stall       = cfg.V_stall,
            t_c_flap      = cfg.t_c_flap,

            high_wing     = cfg.high_wing,
            has_flap_slat = cfg.has_flap_slat,

            rho_bat       = comp["bt"].energy_density,          #kWh/kg Energy density of battery
            rho_fc        = comp["fc_with_hex"].power_density,  #kW/kg Power density of fuel cell system
            rho_ac_dc     = comp["ac_dc"].power_density,        #kW/kg Power density of AC/DC rectifier
            rho_dc_dc_1   = comp["dc_dc_1"].power_density,      #kW/kg Power density of primary DC/DC converter
            rho_dc_dc_2   = comp["dc_dc_2"].power_density,      #kW/kg Power density of secondary DC/DC converter
            rho_dc_ac     = comp["dc_ac"].power_density,        #kW/kg Power density of DC/AC inverter
            rho_cable     = comp["cable"].power_density,        #kW/kg Power density of electrical cables
            rho_pipe      = comp["pipe"].mass_per_length,       #kg/m Mass per length of piping
            rho_turb      = comp["gt_hex"].power_density / cfg.turbine_penalty,    #kW/kg Power density of gas turbine
            rho_HTS_gen   = comp["hts_gen"].power_density / cfg.cryo_penalty,      #kW/kg Power density of HTS generator
            rho_HTS_pow   = comp["hts_pow"].power_density / cfg.cryo_penalty,      #kW/kg Power density of HTS motor
            grav_density  = cfg.grav_density,
            
            P_TO_KW       = P_TO_KW,
            P_max_KW      = P_max_KW,
            W_fuel        = W_fuel,
            configuration = configuration,
            max_Thrust_prop_inner    = max_Thrust_prop_inner,
            max_Thrust_prop_outer    = max_Thrust_prop_outer,

            P_cruise_KW  = P_cruise_KW,
            P_climb_KW   = P_climb_KW,
            P_reserve_KW = P_reserve_KW,
            P_TO_OEI_KW  = P_TO_OEI_KW,
            P_takeoff_KW = P_takeoff_KW,
            P_approach_KW = P_approach_KW,

            t_cruise  = t_cruise,
            t_climb   = t_climb,
            t_reserve = t_reserve,
            N_engines = N_engines,
            bt_charging_ratio = bt_charging_ratio,
            P_opt = P_opt,
            mass_margin = mass_margin,
            N_propellers = cfg.N_propellers,
            aero_parameters = aero_parameters if aero_parameters is not None else {},
            TMS_mass_N2 = cfg.TMS_mass_N2,
            TMS_mass_N4 = cfg.TMS_mass_N4,
            frn_tank_support = cfg.frn_tank_support,
            H2_results_all = H2_results_all
        )


@dataclass
class WeightBreakdown:
    #cooling: dict
    W_wing_initial:   float = 0.0
    W_wing_accurate:  float = 0.0
    W_wing_hld:          float = 0.0
    W_wing_basic:        float = 0.0
    W_htail:  float = 0.0
    W_vtail:  float = 0.0
    W_fus:    float = 0.0
    W_lg:     float = 0.0
    W_lg_nose:  float = 0.0
    W_lg_main: float = 0.0
    W_sc:     float = 0.0
    W_engine: float = 0.0
    W_total_prop: float = 0.0

    # Propulsion breakdown (populated by weightEstimation.compute)
    W_turbine:   float = 0.0
    W_battery:   float = 0.0
    W_fc:        float = 0.0
    W_ac_dc:     float = 0.0
    W_dc_dc:     float = 0.0
    W_dc_ac:     float = 0.0
    W_generator: float = 0.0
    W_motor:     float = 0.0
    W_cable:     float = 0.0
    W_pipe:      float = 0.0
    W_h2_tank:   float = 0.0

    # Power densities and factors stored for display
    rho_turb:    float = 0.0
    rho_bat:     float = 0.0
    rho_fc:      float = 0.0
    rho_HTS_gen: float = 0.0
    rho_HTS_pow: float = 0.0
    rho_ac_dc:    float = 0.0
    rho_dc_dc:    float = 0.0
    rho_dc_ac:    float = 0.0
    rho_cable:    float = 0.0
    rho_pipe:     float = 0.0
    P_TO_KW:     float = 0.0
    P_max_KW:    float = 0.0
    P_cruise_KW: float = 0.0
    P_TO_OEI_KW: float = 0.0
    P_climb_KW:  float = 0.0
    P_reserve_KW: float = 0.0
    P_primary_KW: float = 0.0
    P_secondary_KW: float = 0.0
    max_Thrust_prop_inner: float = 0.0
    max_Thrust_prop_outer: float = 0.0
    W_fuel:      float = 0.0
    mass_breakdown: float = 0.0
    grav_density:float = 0.64
    configuration: int = 1
    total_prop_efficiency: float = 1.0
    climb_eff: float = 1.0
    cruise_eff: float = 1.0
    climb_efficiency: float = 0.0
    cruise_efficiency: float = 0.0
    t_cruise: float = 0.0,
    t_climb  : float = 0.0,
    t_reserve: float = 0.0,
    bt_charging_ratio: float = 0.0,
    P_opt: float = 0.0
    N_propellers: int=1

    H2_results_all: dict = None
    electrical_results: dict = None


    @property
    def W_structure(self) -> float:
        return (self.W_wing_accurate + self.W_htail + self.W_vtail
                + self.W_fus + self.W_lg + self.W_sc)

    @property
    def W_empty(self) -> float:
        return self.W_structure + self.W_total_prop

    def summary(self):
        table = Table(title="Class II Weight Breakdown", show_header=True)
        table.add_column("Group", style="dim")
        table.add_column("Weight (kg)", justify="right")
        table.add_column("Factor / Density", justify="right")

        struct_items = [
            ("Wing_initial",           self.W_wing_initial),
            ("Wing_accurate",     self.W_wing_accurate),
            ("Wing_hld",     self.W_wing_hld),
            ("Wing_basic",     self.W_wing_basic),
            ("Fuselage",       self.W_fus),
            ("Vertical Tail",  self.W_vtail),
            ("Horizontal Tail",self.W_htail),
            ("Landing Gear",   self.W_lg),
            ("Main Landing Gear",   self.W_lg_main),
            ("Nose Landing Gear",   self.W_lg_nose),
            ("Surface Controls", self.W_sc),
        ]

        for name, weight in struct_items:
            table.add_row(name, f"{weight:.1f}", "")

        table.add_section()
        table.add_row("Total Structure", f"[bold]{self.W_structure:.1f}[/bold]", "")

        table.add_section()
        # Propulsion breakdown
        prop_items = [
            (
                "  H2 Tank",
                self.W_h2_tank,
                (f"grav. density = {self.grav_density:.2f}  "
                 f"(W_fuel = {self.W_fuel:.1f} kg)"),
            ),

        ]

        for name, weight, note in prop_items:
            table.add_row(name, f"{weight:.1f}", note)

        table.add_section()
        table.add_row("Open Fan Mass", f"[bold]{self.mass_breakdown["fan"]:.1f}[/bold]", "",)
        table.add_row(
            "Propulsion System without tank",
            f"[bold]{self.W_engine:.1f}[/bold]",
            "",
        )
        table.add_row(
            "Propulsion System with tank (total)",
            f"[bold]{self.W_total_prop:.1f}[/bold]",
            "",
        )


        table.add_section()
        #Power Distribution breakdown
        table.add_row(
            "Cruise Power",
            f"[bold]{self.P_cruise_KW:.1f}[/bold]",
            "",
        )

        table.add_row(
            "OEI Power",
            f"[bold]{self.P_TO_OEI_KW:.1f}[/bold]",
            "",
        )

        table.add_row(
            "Primary Power Unit",
            f"[bold]{self.P_primary_KW:.1f}[/bold]",
            "",
        )

        table.add_row(
            "Secondary Power Unit",
            f"[bold]{self.P_secondary_KW:.1f}[/bold]",
            "",
        )

        table.add_row(
            "Maximum Power required",
            f"[bold]{self.P_max_KW:.1f}[/bold]",
            "",
        )

        table.add_section()
        table.add_row(
            "Power Production Efficiency",
            f"[bold]{self.total_prop_efficiency:.4f}[/bold]",
            "",
        )
        table.add_row(
            "Climb Efficiency",
            f"[bold]{self.climb_efficiency:.4f}[/bold]",
            "",
        )
        table.add_row(
            "Cruise Efficiency",
            f"[bold]{self.cruise_efficiency:.4f}[/bold]",
            "",
        )

        table.add_section()
        table.add_row(
            "[bold green]Empty Weight (excl W_fixed)[/bold green]",
            f"[bold green]{self.W_empty:.1f}[/bold green]",
            "",
        )
        return table


@dataclass
class weightEstimation:

    b_ref = 1.905

    _LG_main = dict(A=18.1, B=0.131, C=0.019, D=2.23e-5)
    _LG_nose = dict(A=9.1,  B=0.082, C=0.0,   D=2.97e-6)

    def __init__(self, geometry: ClassII_Input, comp: dict, write: bool = True):
        self.g = geometry
        self.comp = comp
        self.write = write

    def _validate(self):
        g = self.g
        required = dict(
            MTOW=g.MTOW, MZFW=g.MZFW, b=g.b, S_w=g.S_w, t_r=g.t_r,
            S_h=g.S_h, S_v=g.S_v, b_v=g.b_v,
            b_f=g.b_f, h_f=g.h_f, l_f=g.l_f, l_t=g.l_t,
            V_dive=g.V_dive,
        )
        missing = [k for k, v in required.items() if v <= 0]
        if missing:
            raise ValueError(f"Inputs not set or zero: {missing}")
        if g.MZFW > g.MTOW:
            raise ValueError("Why is your MZFW bigger than MTOW?")

    def _wing_weight_initial(self) -> float:
        g   = self.g
        b_s = g.b * np.cos(g.sweep_half)
        if g.N_propellers>2:
            #change factor for 4 engines on the wing
            return (g.MZFW * g.k_w * b_s**0.75
                * (1 + np.sqrt(self.b_ref / b_s))
                * g.n_ult**0.55
                * ((b_s / g.t_r) / (g.MZFW / g.S_w))**0.3
                * 1.02) *g.mass_margin*0.9
        else:
            #change factor for 2 engines on the wing
            return (g.MZFW * g.k_w * b_s**0.75
                * (1 + np.sqrt(self.b_ref / b_s))
                * g.n_ult**0.55
                * ((b_s / g.t_r) / (g.MZFW / g.S_w))**0.3
                * 1.02) *g.mass_margin*0.95
        
    def _wing_weight_accurate(self) -> float:
        g   = self.g
        aero = g.aero_parameters or {}
        b_s = g.b * np.cos(g.sweep_half)
        k_no=1+np.sqrt(g.b_ref/b_s)
        k_lam=(1+g.taper)**0.4
        W_des=g.MTOW
        W_W_init=self._wing_weight_initial()

        if g.N_propellers>2:
            g.k_e=0.9
            g.k_st=1+g.const_k_st*((g.b*np.cos(g.sweep_LE))**3)/W_des*((g.V_dive/100)/g.t_r)**2*np.cos(g.sweep_half)
            W_basic=(g.const_wing*k_no*k_lam*g.k_e*g.k_uc*g.k_st*\
        ((g.k_b*g.n_ult*(W_des-0.8*W_W_init))**0.55)*\
            (g.b**1.675)*(g.t_r**(-0.45))*np.cos(g.sweep_half)**(-1.325))*1.2
        elif g.N_propellers<=2:
            g.k_e=0.95
            g.k_st=1
            W_basic=g.const_wing*k_no*k_lam*g.k_e*g.k_uc*g.k_st*\
        ((g.k_b*g.n_ult*(W_des-0.8*W_W_init))**0.55)*\
            (g.b**1.675)*(g.t_r**(-0.45))*np.cos(g.sweep_half)**(-1.325)


        
        k_f=g.k_f1*g.k_f2
        required_aero = ["S_f", "b_fs", "delta_defl", "hinge_sweep", "S_LE"]
        missing_aero = [name for name in required_aero if name not in aero]
        if missing_aero:
            raise ValueError(f"Missing aero_parameters entries: {missing_aero}")

        W_hld_TE = (
            g.const_HLD
            * k_f
            * (aero["S_f"] * aero["b_fs"]) ** (3 / 16)
            * ((g.V_stall / 100) ** 2 * (np.sin(aero["delta_defl"]) * np.cos(aero["hinge_sweep"])) / g.t_c_flap) ** (3 / 4)
            * aero["S_f"]
        )
        W_hld_LE = g.W_hld_LE * aero["S_LE"]
        W_hld=W_hld_LE+W_hld_TE


        W_wing_init=W_basic+1.2*(W_hld)
        W_Wing=W_wing_init+0.02*W_wing_init           #adjust for the spoiler and speed brake weights
        return W_Wing,W_hld,W_basic

    def _htail_weight(self) -> float:
        g     = self.g
        S_ft2 = g.S_h * 10.7639
        V_kt  = g.V_dive * 1.94384
        x     = S_ft2**0.2 * V_kt / 1000 / np.sqrt(np.cos(g.sweep_h))
        w_per_area_lb_ft2 = Tail_Interp.get_weight_factor(x)
        return w_per_area_lb_ft2 * S_ft2 * 0.453592 * g.mass_margin

    def _vtail_weight(self) -> float:
        g     = self.g
        S_ft2 = g.S_v * 10.7639
        V_kt  = g.V_dive * 1.94384
        x     = S_ft2**0.2 * V_kt / 1000 / np.sqrt(np.cos(g.sweep_v))
        w_per_area_lb_ft2 = Tail_Interp.get_weight_factor(x)

        if g.t_tail:
            k_v = 1 + 0.15 * g.S_h * g.h_h / (g.S_v * g.b_v)
        else:
            k_v = 1.0
        return w_per_area_lb_ft2 * S_ft2 * k_v * 0.453592 * g.mass_margin

    def _fuselage_weight(self) -> float:
        g       = self.g
        d_eq    = (g.b_f + g.h_f) / 2.0
        sigma   = g.l_f / d_eq
        fact_pres = 1.08                    #torenbeek p.282: add 8% for pressurized
        S_f_wet = (np.pi * g.b_f * g.l_f
                   * (1.0 - 2.0 / sigma)**(2.0 / 3.0)
                   * (1.0 + 1.0 / sigma**2))
        return (g.k_wf
                * np.sqrt(g.V_dive * g.l_t / (g.b_f + g.h_f))
                * S_f_wet ** 1.2) * fact_pres * g.mass_margin

    def _LDG_weight(self) -> float:
        g    = self.g
        k_LG = 1.08 if g.high_wing else 1.0

        def _leg(c: dict) -> float:
            return (c["A"]
                    + c["B"] * g.MTOW**0.75
                    + c["C"] * g.MTOW
                    + c["D"] * g.MTOW**1.5) * g.mass_margin

        return k_LG * _leg(self._LG_main), k_LG * _leg(self._LG_nose)

    def _surface_control_weight(self) -> float:
        g    = self.g
        k_SC = 0.567 if g.has_flap_slat else 0.472
        return 1.2 * k_SC * g.MTOW ** (2 / 3) * g.mass_margin

    def _propulsion_weight(self) -> float:

        #pipe lengths:
        #design A: 82 meters of pipe
        #design B: 34 meters of pipe
        #design C: 34 meters of pipe
        #design D: 48 meters of pipe

        #cable lengths:             #approximated by fuselage length of about 35 meters and wing span of about 28 meters,
                                    #with HTS placed at quarter span
        
        #design A: 36.5 meters of cryo cable      #cable from GT to wing = 1/2 fuselage length + 1/4 wing span + 1/4 wing span, cable from wing to HTS = 1/4 wing span, Battery distance to HTS with 5 meters in total estimated for routing and connections
        #design B: 19 meters of cryo cable      #cable from Battery to wing = 1/4 wing span + 1/4 wing span, Fuel cell distance to HTS with 5 meters in total estimated for routing and connections
        #design C: 5 meters of cryo cable     #Turbine distance to HTS with 5 meters in total estimated for routing and connections
        #design D: 19 meters of cryo cable     #cable from Fuel Cell to wing = 1/4 wing span + 1/4 wing span, Turbine distance to HTS with 5 meters in total estimated for routing and connections

        g     = self.g
        comp = self.comp
        config = g.configuration


        if g.N_propellers>2:
            # component_lists = {
            # 3: {
            #     "components": [
            #         "gt_hex", "gt_hex", "hts_gen", "hts_gen", "ac_dc", "ac_dc",
            #         "dc_ac", "dc_ac","dc_ac","dc_ac", "hts_pow", "hts_pow","hts_pow","hts_pow","open_fan","open_fan","open_fan","open_fan", "cable", "pipe+valves+pumps",
            #     ],
            #     "lengths": {"pipe": 142.0, "cable": 20.0},   #change cable length to 30 meters after iteration to account for more cables needed for DC Bus
            # },
            component_lists = {
            3: {
                "components": [
                    "gt_hex", "gt_hex", "electrical_sys","open_fan","open_fan","open_fan","open_fan", "pipe+valves+pumps",
                ],
                "lengths": {"pipe": 142.0, "cable": 20.0},   #change cable length to 30 meters after iteration to account for more cables needed for DC Bus
            },
        }
            
        else:
            component_lists = { 3: {
                "components": [
                    "gt_hex", "gt_hex", "electrical_sys","open_fan","open_fan", "pipe+valves+pumps",
                ],
                "lengths": {"pipe": 102.0, "cable": 5.0},   #change cable length to 30 meters after iteration to account for more cables needed for DC Bus
            },
            }

        if config not in component_lists:
            raise ValueError(f"Unknown configuration: {config}")

        cfg_data = component_lists[config]
        component_list = cfg_data["components"]
        pipe_len = cfg_data["lengths"]["pipe"]
        cable_len = cfg_data["lengths"]["cable"]
        total_mass = 0.0
        fan_mass = 0.0
        nacelle_factor = 1/0.75 # to account for additional mass of nacelle and integration, estimated as 25% of component mass#
        motor_count=0
        propeller_count=0

        for comp_key in component_list:
            #if comp_key not in comp:
                #raise ValueError(f"Component '{comp_key}' not found in component dict")
            
            if config == 3:
                efficiency = GT_GT_efficiency(comp=comp,t_climb=g.t_climb, t_cruise=g.t_cruise, P_climb=g.P_climb_KW*1000, P_cruise=g.P_cruise_KW*1000)
                efficiency2 = GT_BAT_efficiency(comp=comp,t_climb=g.t_climb, t_cruise=g.t_cruise, P_climb=g.P_climb_KW*1000, P_cruise=g.P_cruise_KW*1000)
                # Similar logic for config 3 but with different component assignments
                # maximum power that flows to the motors (most likely takeoff)
                P_req_tot = max((g.P_cruise_KW), 
                                    g.P_climb_KW, 
                                    g.P_reserve_KW, 
                                    g.P_TO_KW)

                # primary power source requirement is cruise power plus some margin for battery charging or OEI scenario
                P_req_primary = max(g.P_cruise_KW/2, 
                                        g.P_TO_OEI_KW,
                                        P_req_tot/2)

                    # secondary power source requirement is to sustain TO 
                P_req_secondary = max((g.P_TO_KW - P_req_primary), 
                                          g.P_TO_OEI_KW)
                # if comp_key == "cable":
                #     mass = cable_len * comp[comp_key].mass_per_length
                if comp_key == "electrical_sys":
                    electrical_results = perform_complete_electrical_sizing(g.P_takeoff_KW,g.P_climb_KW,g.P_cruise_KW,g.P_approach_KW,
                                                                            g.P_TO_OEI_KW, g.b, write=self.write)
                    if g.N_propellers>2:

                        mass = electrical_results["mass"]
                    else:
                        mass = electrical_results["mass"]*0.9 #estimate that electrical system is 25% lighter for 2 engines since less complex power distribution system and less components needed
                elif comp_key == "pipe+valves+pumps":
                    if g.N_propellers>2:
                        #mass = pipe_len * comp[comp_key].mass_per_length
                        mass = 0
                    else:
                        #mass = pipe_len * comp[comp_key].mass_per_length * 2 #double the mass for 4 engines since more complex piping system with more valves and pumps needed
                        mass = g.TMS_mass_N2                     #big estimate
                elif comp_key == "open_fan" and propeller_count<2 and g.N_propellers>2:
                        mass = g.max_Thrust_prop_inner / comp[comp_key].thrust_density *1.1
                        propeller_count+=1
                        fan_mass+=mass
                elif comp_key == "open_fan" and propeller_count>=2 and g.N_propellers>2:
                        mass = g.max_Thrust_prop_outer / comp[comp_key].thrust_density*1.1
                        fan_mass+=mass
                elif comp_key == "open_fan" and g.N_propellers<=2:
                        mass = g.max_Thrust_prop_inner / comp[comp_key].thrust_density 
                        fan_mass+=mass
                elif comp_key != "electrical_sys" and comp_key != "pipe" and comp_key != "open_fan":
                    pd = comp[comp_key].power_density
                    
                    if comp_key == "gt_hex": #or comp_key == "ac_dc" or comp_key == "hts_gen" or comp_key == "hts_pow" or comp_key == "dc_ac":
                        # mass = P_req_primary / pd / efficiency["GT-MOT_eff"] *nacelle_factor*1.15
                        mass = 0

                    # elif comp_key == "ac_dc":
                    #     mass = P_req_primary / pd / efficiency2["ACDC_eff"]
                    # elif comp_key == "hts_gen":
                    #     mass = P_req_primary / pd / efficiency2["GEN_eff"]
                    # elif comp_key == "hts_pow" and motor_count<2 and g.N_propellers>2:
                    #     mass = P_req_primary / pd *0.8
                    #     motor_count+=1
                    #     fan_mass+=mass
                    # elif comp_key == "hts_pow" and motor_count>=2 and g.N_propellers>2:
                    #     mass = P_req_primary / pd *0.5
                    #     motor_count+=1
                    #     fan_mass+=mass
                    # elif comp_key == "hts_pow" and g.N_propellers<=2:
                    #     mass = P_req_primary / pd
                    
                    # elif comp_key == "dc_ac":
                    #     mass = P_req_primary / pd /efficiency2["Dcac_eff"]
                total_mass += mass

        eff = efficiency["Total_eff"]
        eff_cruise = efficiency["Cruise_average_eff"]
        eff_climb = efficiency["Climb_eff"]
        if "BAT_charging_frac" in efficiency:
            bt_charging_ratio = efficiency["BAT_charging_frac"]
        else:
            bt_charging_ratio = g.bt_charging_ratio

        if "GT_P_opt" in efficiency:
            P_opt = efficiency["GT_P_opt"]/1000
        else:
            P_opt = g.P_opt/1000

        # GT sizing and results
        P_per_flight_condition = [P * 1000 for P in list(electrical_results["powers"]["hts_gen"].values())[0].values()]

        if g.H2_results_all is None:
            H2_temps = [180] * len(P_per_flight_condition)
            H2_press = [25] * len(P_per_flight_condition)
        else:
            H2_temps = []
            H2_press = []
            for condition in ["TO", "climb", "cruise", "APP", "OEI_gt"]:
                H2_temps.append(g.H2_results_all["final_states"][condition]["Temperature_K"])
                H2_press.append(g.H2_results_all["final_states"][condition]["Pressure_Pa"] / 1e5)
            for condition in ["OEI_mot", "OEI_bus"]:
                H2_temps.append(0.5 * (g.H2_results_all["final_states"][condition+"_Working"]["Temperature_K"] + g.H2_results_all["final_states"][condition+"_Failed"]["Temperature_K"]))
                H2_press.append(0.5 * (g.H2_results_all["final_states"][condition+"_Working"]["Pressure_Pa"] + g.H2_results_all["final_states"][condition+"_Failed"]["Pressure_Pa"]) / 1e5)

        gt_results_dict = run_gt_sizing(P_opt=P_opt*1000, off_design_cases=P_per_flight_condition, T_pre_comp=H2_temps, P_pre_comp=H2_press)
        gt_mass = gt_results_dict["dim"].results["m_propulsion"]
        
        mass_flows = [gt_results_dict["od_cases"][P]["mdot_f"] for P in gt_results_dict["od_cases"]]
        gt_effs = [gt_results_dict["od_cases"][P]["eta_total"] for P in gt_results_dict["od_cases"]]
        expander_powers_KW = [gt_results_dict["od_cases"][P]["h2_net_W"]/1000 for P in gt_results_dict["od_cases"]]
        # for P, m in zip(P_per_flight_condition, expander_powers_KW):
        #     print(f"{P}: {m}")

        normal_phases = ['TO', 'climb', 'cruise', 'APP']
        normal_m_dots = [m*2 for m in mass_flows[:4]]
        
        oei_phases = ['OEI_gt', 'OEI_mot', 'OEI_bus']
        oei_m_dots = [mass_flows[4], mass_flows[5]*2, mass_flows[6]*2]

        H2_results_nominal = main_H2_nominal(comps=electrical_results["cooling"], sizes=electrical_results["sizes"],
                                             normal_phases=normal_phases, normal_m_dots=normal_m_dots, write=self.write)
        H2_results_all = main_H2_OEI(comps=electrical_results["cooling"], sizes=electrical_results["sizes"],
                                     All_temps=H2_results_nominal["temperatures"], HEX_areas=H2_results_nominal["areas"], prev_states=H2_results_nominal["final_states"],
                                     oei_phases=oei_phases, oei_m_dots=oei_m_dots, write=self.write)
        # print(H2_results_all)
        TMS_mass, pipe_length = pipe_calculations(b=g.b, sweep_quarter_chord=g.sweep_half) # quarter chord approximated by half for now
        total_mass += TMS_mass + 2 * gt_mass

        mass_breakdown = {
            "gt": gt_mass,
            "mass_flows": mass_flows,
            "expander_powers": expander_powers_KW,
            "efficiencies": gt_effs,
            "TMS": TMS_mass,
            "pipe_length": pipe_length,
            "electrical": electrical_results["mass"],
            "fan": fan_mass,
        }

        return total_mass * g.mass_margin, mass_breakdown, P_req_primary, P_req_secondary, P_req_tot,eff,eff_climb, \
            eff_cruise, bt_charging_ratio, P_opt, electrical_results, H2_results_all
    
    def _h2_tank_weight(self) -> float:
        # return (self.g.W_fuel * (1 / self.g.grav_density - 1)*(1+self.g.frn_tank_support))* self.g.mass_margin
        m_tank, _, _ = main_storage(self.g.W_fuel)
        # print(m_tank)
        return m_tank

    def compute(self) -> WeightBreakdown:
        self._validate()
        g = self.g

        h2_tank_weight   = self._h2_tank_weight()
        W_engine_total, mass_breakdown, P_req_primary, P_req_secondary, P_req_tot,\
            total_prop_efficiency, climb_eff,cruise_eff, bt_charging_ratio, P_opt, electrical_results, H2_results_all = self._propulsion_weight()
        W_wing_accurate, W_hld, W_basic = self._wing_weight_accurate()
        
        W_lg_main, W_lg_nose = self._LDG_weight()
        W_lg = W_lg_main + W_lg_nose

        return WeightBreakdown(
            W_wing_initial   = self._wing_weight_initial(),
            W_wing_accurate=W_wing_accurate,
            W_wing_hld = W_hld,
            W_wing_basic = W_basic,
            W_htail  = self._htail_weight(),
            W_vtail  = self._vtail_weight(),
            W_fus    = self._fuselage_weight(),
            W_lg     = W_lg,
            W_lg_main = W_lg_main,
            W_lg_nose = W_lg_nose,
            W_sc     = self._surface_control_weight(),
            W_engine = W_engine_total,
            W_total_prop = W_engine_total + h2_tank_weight,
            mass_breakdown = mass_breakdown,

            # Propulsion detail
            W_h2_tank   = h2_tank_weight,

            # For display in summary
            grav_density = g.grav_density,
            P_TO_KW      = g.P_TO_KW,
            W_fuel       = g.W_fuel,

            #power values for display
            P_cruise_KW = g.P_cruise_KW,
            P_climb_KW = g.P_climb_KW,
            P_TO_OEI_KW  = g.P_TO_OEI_KW,
            P_primary_KW = P_req_primary,
            P_secondary_KW = P_req_secondary,
            P_max_KW=  P_req_tot,
            total_prop_efficiency = total_prop_efficiency,
            climb_eff = climb_eff,
            cruise_eff = cruise_eff,
            climb_efficiency=climb_eff,
            cruise_efficiency=cruise_eff,
            bt_charging_ratio=bt_charging_ratio,
            P_opt=P_opt,
            electrical_results=electrical_results,

            H2_results_all=H2_results_all
        )


    def iterate_MTOW(
        self,
        W_payload:         float,
        W_fuel:            float,
        W_fixed_equipment: float = 0.0,
        tol:               float = 1.0,
        max_iter:          int   = 50,
    ) -> tuple[float, WeightBreakdown]:
        """
        Standalone weight-only MTOW iteration. Kept for backward compatibility
        with your original main. The full Class II loop lives in main.py.

        Bug fix vs original: MZFW is now computed correctly as
            MZFW = W_empty + W_payload + W_fixed_equipment
        rather than (MTOW - W_fuel) which was only correct at convergence.
        """
        bd = WeightBreakdown()
        for i in range(max_iter):
            bd        = self.compute()
            MZFW_new  = bd.W_empty + W_payload + W_fixed_equipment
            MTOW_new  = MZFW_new + W_fuel
            delta     = abs(MTOW_new - self.g.MTOW)

            self.g.MTOW = MTOW_new
            self.g.MZFW = MZFW_new

            if delta < tol:
                print(f"Converged in {i + 1} iterations.  MTOW = {MTOW_new:.1f} kg")
                return MTOW_new, bd

        print(f"Warning: did not converge after {max_iter} iterations. "
              f"Residual = {delta:.2f} kg")
        return self.g.MTOW, bd


if __name__ == "__main__":
    from Aircraft_Config import default_q400_hycool

    comp = comp_params
    cfg = default_q400_hycool()
    geo = ClassII_Input.from_config(cfg, comp, base_params=True)
    est = weightEstimation(geo)

    print("--- Single-shot ---")
    print(est.compute().summary())

    print("\n--- Iterated MTOW (weight-only loop) ---")
    _, bd = est.iterate_MTOW(
        W_payload         = cfg.W_payload,
        W_fuel            = 600.0,
        W_fixed_equipment = cfg.W_fixed,
    )
    print(bd.summary())