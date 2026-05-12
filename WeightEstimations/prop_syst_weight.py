import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from General.component_parameters import component_params as comp_params
from Aircraft_Config   import AircraftConfig, default_q400_hycool
from mainClassII import ClassIIResult, run_class_ii
from dataclasses import dataclass

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns


@dataclass
class PropulsionUnitWeight:

    MTOW:        float
    P_cruise:        float
    P_climb:     float
    P_reserve:      float
    P_TO:       float
    P_TO_OEI:   float
    W_power:     float
    iterations:  int
    converged:   bool

    classII:        ClassIIResult
    components:       dict


    def summary(self):
        status_color = "green" if self.converged else "red"
        main_info = (
            f"MTOW: {self.MTOW/1000:.2f} t\n"
            f"Cruise Power:  {self.P_cruise/1000:.2f} MW\n"
            f"Climb Power: {self.P_climb/1000:.2f} MW\n"
            f"Reserve Power: {self.P_reserve/1000:.2f} MW\n"
            f"Take off Power: {self.P_TO/1000:.2f} MW\n"
            f"TO/OEI Power: {self.P_TO_OEI/1000:.2f} MW\n"
            f"Power Unit Mass: {self.W_power:.2f} kg\n"
            f"Iterations: {self.iterations}"
        )

        return Panel(
            Columns([main_info]),
            title=f"[bold {status_color}]Power Unit Calculations[/bold {status_color}]",
            border_style=status_color
        )
    
def calculate_power_unit_weight(
    cfg:      ClassIIResult,
    comp:     dict,
    config:      int,
) -> PropulsionUnitWeight:
    
    # Compute power requirements based on Class II results
    P_cruise = cfg.mission.P_cruise_shaft / 1e3     # kW
    P_climb = cfg.mission.P_climb_shaft / 1e3        # kW
    P_reserve = cfg.mission.P_reserve_shaft / 1e3    # kW
    P_TO_OEI = cfg.power.P_TO_per_engine / 1e3       # kW
    P_TO = cfg.power.P_TO_total / 1e3                      # kW

    #Compute time requirements based on Class II results
    t_cruise = cfg.mission.t_cruise  # s
    t_climb = cfg.mission.t_climb  # s
    t_reserve = cfg.mission.t_reserve  # s


    PD_GT       =   comp["gt"].power_density  #kW/kg Power density of gas turbine
    PD_FC_syst  =   comp["fc"].power_density  #kW/kg Power density of fuel cell system
    PD_HTS      =   comp["hts"].power_density  #kW/kg Power density of HTS motor
    PD_GT_HEX   =   comp["gt_hex"].power_density  #kW/kg Power density of gas turbine with heat exchanger
    PD_DCDC     =   comp["dc_dc"].power_density  #kW/kg Power density of DC/DC converter
    PD_ACDC     =   comp["ac_dc"].power_density  #kW/kg Power density of AC/DC rectifier
    PD_DCAC     =   comp["dc_ac"].power_density  #kW/kg Power density of DC/AC inverter
    ED_BATT     =   comp["bt"].energy_density  #kWh/kg Energy density of battery
    PD_CABLE    =   comp["cable"].power_density  #kW/kg Power density of electrical cables

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


    if config == 1:
        component_list = ["gt_hex", "bt", "hts_gen", "ac_dc","dc_dc", "dc_ac","hts_pow","hts_pow", "cable","pipe"]
    elif config == 2:
        component_list = ["fc", "bt", "dc_dc", "dc_dc", "dc_ac", "hts_pow","hts_pow", "cable","pipe"]
    elif config == 3:
        component_list = ["gt_hex", "gt_hex", "hts_gen", "hts_gen", "ac_dc","ac_dc", "dc_ac","dc_ac","hts_pow","hts_pow", "cable","pipe"]
    elif config == 4:
        component_list = ["gt_hex", "gt_hex", "hts_gen", "hts_gen", "fc", "ac_dc", "ac_dc", "dc_dc", "dc_ac", "dc_ac", "hts_pow","hts_pow", "cable","pipe"]
    else:
        raise ValueError(f"Unknown configuration: {config}")

    # Compute total mass: convert P (MW) -> kW, mass = P_kW / power_density (kW/kg)
    # P_req_MW = cfg.mission.P_climb_shaft / 1e6
    # P_req_kW = P_req_MW * 1000.0

    total_mass = 0.0

    for comp_key in component_list:
        if comp_key not in comp:
            raise ValueError(f"Component '{comp_key}' not found in component dict")
        elif config == 1:
            #5% of cruise power but put this in some input file!
            bt_charging_ratio = 0.05 
            pd = comp[comp_key].power_density
            #maximum power that flows to the motors (most likely takeoff)
            P_req_tot = max((P_cruise*(1+bt_charging_ratio)), P_climb, P_reserve, P_TO)
            #primary power generator power requirement is cruise power plus some margin for battery charging
            P_req_primary = P_cruise*(1+bt_charging_ratio)  
            P_req_secondary = max((P_climb - P_req_primary), P_TO_OEI)
            if comp_key == "gt_hex" or comp_key == "hts_gen" or comp_key == "ac_dc":
                mass = P_req_primary / pd
            elif comp_key == "bt":
                energy_required_kWh = P_req_secondary * (t_climb / 3600)  # Convert seconds to hours
                ed = comp[comp_key].energy_density
                mass = max(energy_required_kWh / ed, P_req_secondary / pd)
            elif comp_key == "dc_dc":
                mass = P_req_secondary / pd
            elif comp_key == "dc_ac": 

            



        pd = comp[comp_key].power_density
        total_mass += mass

    # Build and return a PropulsionUnitWeight
    return PropulsionUnitWeight(
        MTOW=cfg.MTOW,
        P_cruise=P_cruise/1e3,  # MW
        P_climb=P_climb/1e3,    # MW
        P_reserve=P_reserve/1e3,
        P_TO_OEI=P_TO_OEI/1e3,
        P_TO=P_TO/1e3,  # MW
        W_power=total_mass,
        iterations=0,
        converged=True,
        classII=cfg,
        components=comp,
    )

def run_Power_sizing(
    cfg:      AircraftConfig,
    comp:     dict,
    tol:      float = 50,
    max_iter: int   = 10,
    verbose:  bool  = True,
) -> PropulsionUnitWeight:

    # Run Class II, then size power unit, and update propulsion mass
    # into cfg.W_fixed and repeat until propulsion mass change < tol (kg).
    base_W_fixed = cfg.W_fixed
    prev_W_power = None
    converged = False

    config = int(input("Enter design configuration (1-4): "))

    for it in range(1, max_iter + 1):
        # 1) Run Class II with current config
        result = run_class_ii(cfg, verbose=False)

        # 2) Size propulsion unit based on Class II result
        pw = calculate_power_unit_weight(result, comp, config)
        W_power = pw.W_power

        # 3) Inject propulsion mass into configuration and prepare next iteration
        cfg.W_fixed = base_W_fixed + W_power

        if verbose:
            print(f"iter {it}: W_power={W_power:.1f} kg, cfg.W_fixed={cfg.W_fixed:.1f} kg")

        # 4) Check convergence on propulsion mass change
        if prev_W_power is not None and abs(W_power - prev_W_power) < tol:
            converged = True
            pw.iterations = it
            pw.converged = True
            if verbose:
                print(f"Converged after {it} iterations (ΔW={abs(W_power-prev_W_power):.2f} kg)")
            return pw

        prev_W_power = W_power

    # If we reach here, did not converge within max_iter; return last result
    pw.iterations = max_iter
    pw.converged = False
    if verbose:
        print(f"Warning: power sizing did not converge after {max_iter} iterations")
    return pw


#possibly add reserve OEI if required
# P_bat_climb = P_climb - P_cruise
# P_bat_OEI = P_TO_OEI
# P_bat_req = max(P_bat_climb, P_bat_OEI)


if __name__ == "__main__":
    cfg = default_q400_hycool()
    result = run_Power_sizing(cfg=cfg, comp=comp_params, tol=1.0, max_iter=10, verbose=True)
    print(result.summary())