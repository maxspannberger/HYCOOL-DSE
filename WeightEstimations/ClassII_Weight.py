import numpy as np
from dataclasses import dataclass
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import Tail_Interpolation as Tail_Interp
from Aircraft_Config import AircraftConfig
from General.component_parameters import component_params as comp_params

from rich.table import Table



@dataclass
class ClassII_Input:
    # Weights
    MTOW:       float = 0.0
    MZFW:       float = 0.0
    n_ult:      float = 3.75

    # Wing
    b:          float = 0.0
    S_w:        float = 0.0
    sweep_half: float = 0.0
    t_r:        float = 0.0
    k_w:        float = 6.67e-3

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
    W_fuel:      float = 0.0
    grav_density:float = 0.64
    configuration: int = 1
    cable_lentgh: float = 0.0
    pipe_length:  float = 0.0
    N_engines: float = 0.0

    #flight phase times
    t_cruise: float = 0.0
    t_climb: float = 0.0
    t_reserve: float = 0.0



    @classmethod
    def from_config(
        cls,
        cfg: AircraftConfig,
        comp: dict,
        MTOW: Optional[float] = None,
        MZFW: Optional[float] = None,
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
        t_cruise: float = 0.0,
        t_climb: float = 0.0,
        t_reserve: float = 0.0,
        N_engines: float = 0.0,
    ) -> "ClassII_Input":
        """
        Build the weight-estimator input from a shared AircraftConfig.

        S_h, S_v, b_v default to the initial guesses in cfg but should be
        overridden by the tail-sizing module's outputs each iteration.
        MTOW defaults to cfg.MTOW_initial.
        """
        return cls(
            MTOW          = MTOW if MTOW is not None else cfg.MTOW_initial,
            MZFW          = MZFW if MZFW is not None else cfg.MTOW_initial * 0.95,
            n_ult         = cfg.n_ult,

            b             = cfg.b,
            S_w           = cfg.S_ref,
            sweep_half    = cfg.sweep_half,
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

            high_wing     = cfg.high_wing,
            has_flap_slat = cfg.has_flap_slat,

            rho_bat     =   comp["bt"].energy_density,  #kWh/kg Energy density of battery
            rho_fc      =   comp["fc_with_hex"].power_density,  #kW/kg Power density of fuel cell system
            rho_ac_dc   =   comp["ac_dc"].power_density,  #kW/kg Power density of AC/DC rectifier
            rho_dc_dc_1   =   comp["dc_dc_1"].power_density,  #kW/kg Power density of primary DC/DC converter
            rho_dc_dc_2   =   comp["dc_dc_2"].power_density,  #kW/kg Power density of secondary DC/DC converter
            rho_dc_ac   =   comp["dc_ac"].power_density,  #kW/kg Power density of DC/AC inverter
            rho_cable   =   comp["cable"].power_density,  #kW/kg Power density of electrical cables
            rho_pipe    =   comp["pipe"].mass_per_length,  #kg/m Mass per length of piping


            rho_turb      = comp["gt_hex"].power_density / cfg.turbine_penalty,     #kW/kg Power density of gas turbine
            rho_HTS_gen       = comp["hts_gen"].power_density / cfg.cryo_penalty,           #kW/kg Power density of HTS generator
            rho_HTS_pow       = comp["hts_pow"].power_density / cfg.cryo_penalty,           #kW/kg Power density of HTS motor
            grav_density  = cfg.grav_density,
            
            P_TO_KW       = P_TO_KW,
            P_max_KW      = P_max_KW,
            W_fuel        = W_fuel,
            configuration = configuration,

            P_cruise_KW  = P_cruise_KW,
            P_climb_KW   = P_climb_KW,
            P_reserve_KW = P_reserve_KW,
            P_TO_OEI_KW  = P_TO_OEI_KW,

            t_cruise  = t_cruise,
            t_climb   = t_climb,
            t_reserve = t_reserve,
            N_engines = N_engines,
        )


@dataclass
class WeightBreakdown:
    W_wing:   float = 0.0
    W_htail:  float = 0.0
    W_vtail:  float = 0.0
    W_fus:    float = 0.0
    W_lg:     float = 0.0
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
    W_fuel:      float = 0.0
    grav_density:float = 0.64
    configuration: int = 1


    @property
    def W_structure(self) -> float:
        return (self.W_wing + self.W_htail + self.W_vtail
                + self.W_fus + self.W_lg + self.W_sc)

    @property
    def W_empty(self) -> float:
        return self.W_structure + self.W_engine

    def summary(self):
        table = Table(title="Class II Weight Breakdown", show_header=True)
        table.add_column("Group", style="dim")
        table.add_column("Weight (kg)", justify="right")
        table.add_column("Factor / Density", justify="right")

        struct_items = [
            ("Wing",           self.W_wing),
            ("Fuselage",       self.W_fus),
            ("Vertical Tail",  self.W_vtail),
            ("Horizontal Tail",self.W_htail),
            ("Landing Gear",   self.W_lg),
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
        table.add_row(
            "[bold green]OEW (Empty Weight)[/bold green]",
            f"[bold green]{self.W_empty:.1f}[/bold green]",
            "",
        )
        return table


@dataclass
class weightEstimation:

    b_ref = 1.905

    _LG_main = dict(A=18.1, B=0.131, C=0.019, D=2.23e-5)
    _LG_nose = dict(A=9.1,  B=0.082, C=0.0,   D=2.97e-6)

    def __init__(self, geometry: ClassII_Input, comp: dict = comp_params):
        self.g = geometry
        self.comp = comp

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

    def _wing_weight(self) -> float:
        g   = self.g
        b_s = g.b * np.cos(g.sweep_half)
        return (g.MZFW * g.k_w * b_s**0.75
                * (1 + np.sqrt(self.b_ref / b_s))
                * g.n_ult**0.55
                * ((b_s / g.t_r) / (g.MZFW / g.S_w))**0.3
                * 1.02)

    def _htail_weight(self) -> float:
        g     = self.g
        S_ft2 = g.S_h * 10.7639
        V_kt  = g.V_dive * 1.94384
        x     = S_ft2**0.2 * V_kt / 1000 / np.sqrt(np.cos(g.sweep_h))
        w_per_area_lb_ft2 = Tail_Interp.get_weight_factor(x)
        return w_per_area_lb_ft2 * S_ft2 * 0.453592

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
        return w_per_area_lb_ft2 * S_ft2 * k_v * 0.453592

    def _fuselage_weight(self) -> float:
        g       = self.g
        d_eq    = (g.b_f + g.h_f) / 2.0
        sigma   = g.l_f / d_eq
        S_f_wet = (np.pi * g.b_f * g.l_f
                   * (1.0 - 2.0 / sigma)**(2.0 / 3.0)
                   * (1.0 + 1.0 / sigma**2))
        return (g.k_wf
                * np.sqrt(g.V_dive * g.l_t / (g.b_f + g.h_f))
                * S_f_wet ** 1.2)

    def _LDG_weight(self) -> float:
        g    = self.g
        k_LG = 1.08 if g.high_wing else 1.0

        def _leg(c: dict) -> float:
            return (c["A"]
                    + c["B"] * g.MTOW**0.75
                    + c["C"] * g.MTOW
                    + c["D"] * g.MTOW**1.5)

        return k_LG * (_leg(self._LG_main) + _leg(self._LG_nose))

    def _surface_control_weight(self) -> float:
        g    = self.g
        k_SC = 0.567 if g.has_flap_slat else 0.472
        return 1.2 * k_SC * g.MTOW ** (2 / 3)

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

        component_lists = {
            1: {
                "components": [
                    "gt_hex", "bt", "hts_gen", "ac_dc", "dc_dc_2", "dc_ac",
                    "hts_pow", "hts_pow", "cable", "pipe",
                ],
                "lengths": {"pipe": 82.0, "cable": 36.5},
            },
            2: {
                "components": [
                    "fc_with_hex", "bt", "dc_dc_1", "dc_dc_2", "dc_ac",
                    "hts_pow", "hts_pow", "cable", "pipe",
                ],
                "lengths": {"pipe": 34.0, "cable": 19.0},
            },
            3: {
                "components": [
                    "gt_hex", "gt_hex", "hts_gen", "hts_gen", "ac_dc", "ac_dc",
                    "dc_ac", "dc_ac", "hts_pow", "hts_pow", "cable", "pipe",
                ],
                "lengths": {"pipe": 34.0, "cable": 5.0},
            },
            4: {
                "components": [
                    "gt_hex", "gt_hex", "hts_gen", "hts_gen", "fc_with_hex", "ac_dc",
                    "ac_dc", "dc_dc_2", "dc_ac", "dc_ac", "hts_pow", "hts_pow", "cable", "pipe",
                ],
                "lengths": {"pipe": 48.0, "cable": 19.0},
            },
        }

        if config not in component_lists:
            raise ValueError(f"Unknown configuration: {config}")

        cfg_data = component_lists[config]
        component_list = cfg_data["components"]
        pipe_len = cfg_data["lengths"]["pipe"]
        cable_len = cfg_data["lengths"]["cable"]
        total_mass = 0.0

        for comp_key in component_list:
            if comp_key not in comp:
                raise ValueError(f"Component '{comp_key}' not found in component dict")
            elif config == 1:

                if comp_key == "cable":
                    mass = cable_len * comp[comp_key].mass_per_length
                elif comp_key == "pipe":
                    mass = pipe_len * comp[comp_key].mass_per_length   
                elif comp_key != "cable" and comp_key != "pipe":
                    # 5% of cruise power but put this in some input file!
                    bt_charging_ratio = 0.05 
                    pd = comp[comp_key].power_density

                    # maximum power that flows to the motors (most likely takeoff)
                    P_req_tot = max((g.P_cruise_KW*(1+bt_charging_ratio)), g.P_climb_KW, g.P_reserve_KW, g.P_TO_KW)

                    # primary power source requirement is cruise power plus some margin for battery charging or OEI scenario
                    P_req_primary = max(g.P_cruise_KW*(1+bt_charging_ratio), g.P_TO_OEI_KW)
                    
                    # secondary power source requirement is to sustain TO 
                    P_req_secondary = max((g.P_TO_KW - P_req_primary), g.P_TO_OEI_KW)
                    if comp_key == "gt_hex" or comp_key == "hts_gen" or comp_key == "ac_dc":
                        mass = P_req_primary / pd
                    elif comp_key == "bt":
                        energy_required_kWh = P_req_secondary * (g.t_climb / 3600)  # Convert seconds to hours
                        ed = comp[comp_key].energy_density
                        mass = max(energy_required_kWh / ed, P_req_secondary / pd)
                    elif comp_key == "dc_dc_2":
                        mass = P_req_secondary / pd
                    elif comp_key == "dc_ac" or comp_key == "hts_pow":
                        max_P_per_string = max(P_req_tot/2, g.P_TO_OEI_KW)
                        mass = max_P_per_string / pd      
                total_mass += mass

            elif config == 2:
                # Similar logic for config 2 but with fuel cell instead of gas turbine
                if comp_key == "cable":
                    mass = cable_len * comp[comp_key].mass_per_length
                elif comp_key == "pipe":
                    mass = pipe_len * comp[comp_key].mass_per_length   
                elif comp_key != "cable" and comp_key != "pipe":
                    # 5% of cruise power but put this in some input file!
                    bt_charging_ratio = 0.05 
                    pd = comp[comp_key].power_density
                    # maximum power that flows to the motors (most likely takeoff)
                    P_req_tot = max((g.P_cruise_KW*(1+bt_charging_ratio)), g.P_climb_KW, g.P_reserve_KW, g.P_TO_KW)
                    # primary power source requirement is cruise power plus some margin for battery charging or OEI scenario
                    P_req_primary = max(g.P_cruise_KW*(1+bt_charging_ratio), g.P_TO_OEI_KW)
                    # secondary power source requirement is to sustain TO 
                    P_req_secondary = max((g.P_TO_KW - P_req_primary), (g.P_TO_OEI_KW-(1/2)*P_req_primary))
                    if comp_key == "fc_with_hex" or comp_key == "dc_dc_1":
                        mass = P_req_primary / pd
                    elif comp_key == "bt":
                        energy_required_kWh = P_req_secondary * (g.t_climb / 3600)  # Convert seconds to hours
                        ed = comp[comp_key].energy_density
                        mass = max(energy_required_kWh / ed, P_req_secondary / pd)
                    elif comp_key == "dc_dc_2":
                        mass = P_req_secondary / pd
                    elif comp_key == "dc_ac" or comp_key == "hts_pow":
                        max_P_per_string = max(P_req_tot/2, g.P_TO_OEI_KW)
                        mass = max_P_per_string / pd
                total_mass += mass

            
            elif config == 3:
                # Similar logic for config 3 but with different component assignments
                if comp_key == "cable":
                    mass = cable_len * comp[comp_key].mass_per_length
                elif comp_key == "pipe":
                    mass = pipe_len * comp[comp_key].mass_per_length   
                elif comp_key != "cable" and comp_key != "pipe":
                    pd = comp[comp_key].power_density
                    # maximum power that flows to the motors (most likely takeoff)
                    P_req_tot = max((g.P_cruise_KW), g.P_climb_KW, g.P_reserve_KW, g.P_TO_KW)
                    # primary power source requirement is cruise power plus some margin for battery charging or OEI scenario
                    P_req_primary = max(g.P_cruise_KW/2, g.P_TO_OEI_KW,P_req_tot/2)
                    # secondary power source requirement is to sustain TO 
                    P_req_secondary = max((g.P_TO_KW - P_req_primary), g.P_TO_OEI_KW)
                    if comp_key == "gt_hex" or comp_key == "ac_dc" or comp_key == "hts_gen" or comp_key == "hts_pow" or comp_key == "dc_ac":
                        mass = P_req_primary / pd
                total_mass += mass

            elif config == 4:
                # Similar logic for config 4 but with different component assignments 
                if comp_key == "cable":
                    mass = cable_len * comp[comp_key].mass_per_length
                elif comp_key == "pipe":
                    mass = pipe_len * comp[comp_key].mass_per_length   
                elif comp_key != "cable" and comp_key != "pipe":
                    pd = comp[comp_key].power_density
                    # maximum power that flows to the motors (most likely takeoff)
                    P_req_tot = max(g.P_cruise_KW, g.P_climb_KW, g.P_reserve_KW, g.P_TO_KW)
                    # primary power source requirement is cruise power or OEI scenario, GT's together must suffice in both situations
                    P_req_primary = max(g.P_cruise_KW, g.P_TO_OEI_KW)
                    # secondary power source requirement is to sustain TO 
                    P_req_secondary = max((P_req_tot - P_req_primary), (g.P_TO_OEI_KW-(1/2)*P_req_primary))
                    if comp_key == "gt_hex" or comp_key == "hts_gen" or comp_key == "ac_dc":
                        mass = (P_req_primary/2) / pd
                    elif comp_key == "fc_with_hex" or comp_key == "dc_dc_2":
                        mass = P_req_secondary / pd
                    elif comp_key == "dc_ac" or comp_key == "hts_pow":
                        max_P_per_string = max(P_req_tot/2, g.P_TO_OEI_KW)
                        mass = max_P_per_string / pd
                total_mass += mass

        return total_mass
    
    def _h2_tank_weight(self) -> float:
        return self.g.W_fuel * (1 / self.g.grav_density - 1)

    def compute(self) -> WeightBreakdown:
        self._validate()
        g = self.g

        h2_tank_weight   = self._h2_tank_weight()
        W_engine_total   = self._propulsion_weight()

        return WeightBreakdown(
            W_wing   = self._wing_weight(),
            W_htail  = self._htail_weight(),
            W_vtail  = self._vtail_weight(),
            W_fus    = self._fuselage_weight(),
            W_lg     = self._LDG_weight(),
            W_sc     = self._surface_control_weight(),
            W_engine = W_engine_total,
            W_total_prop = W_engine_total + h2_tank_weight,

            # Propulsion detail
            W_h2_tank   = h2_tank_weight,

            # For display in summary
            grav_density = g.grav_density,
            P_TO_KW      = g.P_TO_KW,
            W_fuel       = g.W_fuel,
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
    geo = ClassII_Input.from_config(cfg, comp)
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