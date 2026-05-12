import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from General.component_parameters import Component
from Aircraft_Config   import AircraftConfig, default_q400_hycool
from mainClassII import ClassIIResult
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
    P_TO_OEI:   float
    W_power:     float
    iterations:  int
    converged:   bool

    classII:        ClassIIResult
    components:       Component


    def summary(self):
        status_color = "green" if self.converged else "red"
        main_info = (
            f"MTOW: {self.MTOW/1000:.2f} t\n"
            f"Cruise Power:  {self.P_cruise:.2f} MW\n"
            f"Climb Power: {self.P_climb:.2f} MW\n"
            f"Reserve Power: {self.P_reserve:.2f} MW\n"
            f"TO/OEI Power: {self.P_TO_OEI:.2f} MW\n"
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
    comp:     Component,
    tol:      float = 1.0,
    max_iter: int   = 100,
    verbose:  bool  = True,
) -> PropulsionUnitWeight:
    # This function will implement the logic to calculate the propulsion unit weight based on the power requirements from the Class II results and the component parameters.
    # The implementation will depend on the specific details of how you want to model the power unit mass based on the power requirements and component characteristics.
    pass

def run_Power_sizing(
    cfg:      ClassIIResult,
    comp:     Component,
    tol:      float = 1.0,
    max_iter: int   = 100,
    verbose:  bool  = True,
) -> PropulsionUnitWeight:


    # -----------------------------------------------------------------
    # Step 1: define first power unit sizing outside of class II estimation
    # -----------------------------------------------------------------
    MTOW    = cfg.MTOW
    converged = False
    it = 0

    for it in range(1,max_iter+1):
        # Compute power requirements based on Class II results
        P_cruise = cfg.mission.P_cruise_shaft / 1e6  # MW
        P_climb = cfg.mission.P_climb_shaft / 1e6    # MW
        P_reserve = cfg.mission.P_reserve_shaft / 1e6  # MW
        P_TO_OEI = cfg.power.P_TO_per_engine / 1e6   # MW

        #Compute time requirements based on Class II results
        t_cruise = cfg.mission.t_cruise  # s
        t_climb = cfg.mission.t_climb  # s
        t_reserve = cfg.mission.t_reserve  # s

        # Here you would implement your logic to compute W_power based on the power requirements and component parameters
        W_power = PropulsionUnitWeight(MTOW, P_cruise, P_climb, P_reserve, P_TO_OEI, 0, 0, False)

        # Check for convergence (this is a placeholder, replace with actual logic)
        if abs(W_power - cfg.weight.W_fixed) < tol:
            converged = True
            break

# Run Class II once and reuse the config/result
cfg = default_q400_hycool()
class_ii_result = run_class_ii(cfg, verbose=False)

#--------------------- INPUTS ------------------------- (these will be changed to take proper variables from other files)

P_cruise    =   class_ii_result.mission.P_cruise_shaft/ 1e6  #MW 
P_climb     =   class_ii_result.mission.P_climb_shaft/ 1e6  #MW
P_reserve   =   class_ii_result.mission.P_reserve_shaft/ 1e6  #MW

t_cruise    =   class_ii_result.mission.t_cruise #s
t_climb     =   class_ii_result.mission.t_climb  #s
t_reserve   =   class_ii_result.mission.t_reserve #s

# Use per-engine takeoff power (watts) and compute OEI power in MW
P_TO_OEI =  class_ii_result.power.P_TO_per_engine/ 1e6  # MW
print(f"Per-engine takeoff power: {P_TO_OEI:.2f} MW")
print(f"Climb power: {P_climb:.2f} MW")
print(f"Cruise power: {P_cruise:.2f} MW")
print(f"Reserve power: {P_reserve:.2f} MW")

#--------------------- Power/energy densities -------------------------
PD_GT       =   c["gt"].power_density  #kW/kg Power density of gas turbine
PD_FC_syst  =   c["fc"].power_density  #kW/kg Power density of fuel cell system
PD_HTS      =   c["hts"].power_density  #kW/kg Power density of HTS motor
PD_GT_HEX   =   c["gt_hex"].power_density  #kW/kg Power density of gas turbine with heat exchanger
PD_DCDC     =   c["dc_dc"].power_density  #kW/kg Power density of DC/DC converter
PD_ACDC     =   c["ac_dc"].power_density  #kW/kg Power density of AC/DC rectifier
PD_DCAC     =   c["dc_ac"].power_density  #kW/kg Power density of DC/AC inverter
ED_BATT     =   c["bt"].energy_density  #kWh/kg Energy density of battery
PD_CABLE    =   c["cable"].power_density  #kW/kg Power density of electrical cables

#--------------------- CALCULATIONS -------------------------
#pipe lengths:
#design A: 82 meters of pipe
#design B: 34 meters of pipe
#design C: 34 meters of pipe
#design D: 48 meters of pipe


#possibly add reserve OEI if required
P_bat_climb = P_climb - P_cruise
P_bat_OEI = P_TO_OEI
P_bat_req = max(P_bat_climb, P_bat_OEI)

def Design_1_mass(component_list, P_req):
    total_mass = 0
    for comp in component_list:
        if comp == "gt":
            mass = P_req / PD_GT
        elif comp == "fc":
            mass = P_req / PD_FC_syst
        elif comp == "hts":
            mass = P_req / PD_HTS
        elif comp == "gt_hex":
            mass = P_req / PD_GT_HEX
        elif comp == "dc_dc":
            mass = P_req / PD_DCDC
        elif comp == "ac_dc":
            mass = P_req / PD_ACDC
        elif comp == "dc_ac":
            mass = P_req / PD_DCAC
        else:
            raise ValueError(f"Unknown component: {comp}")
        total_mass += mass

        return total_mass

def Design_2_mass(component_list, P_req):
    total_mass = 0
    for comp in component_list:
        if comp == "gt":
            mass = P_req / PD_GT
        elif comp == "fc":
            mass = P_req / PD_FC_syst
        elif comp == "hts":
            mass = P_req / PD_HTS
        elif comp == "gt_hex":
            mass = P_req / PD_GT_HEX
        elif comp == "dc_dc":
            mass = P_req / PD_DCDC
        elif comp == "ac_dc":
            mass = P_req / PD_ACDC
        elif comp == "dc_ac":
            mass = P_req / PD_DCAC
        else:
            raise ValueError(f"Unknown component: {comp}")
        total_mass += mass

        return total_mass

def Design_3_mass(component_list, P_req):
    total_mass = 0
    for comp in component_list:
        if comp == "gt":
            mass = P_req / PD_GT
        elif comp == "fc":
            mass = P_req / PD_FC_syst
        elif comp == "hts":
            mass = P_req / PD_HTS
        elif comp == "gt_hex":
            mass = P_req / PD_GT_HEX
        elif comp == "dc_dc":
            mass = P_req / PD_DCDC
        elif comp == "ac_dc":
            mass = P_req / PD_ACDC
        elif comp == "dc_ac":
            mass = P_req / PD_DCAC
        else:
            raise ValueError(f"Unknown component: {comp}")
        total_mass += mass
        return total_mass
    
def Design_4_mass(component_list, P_req):
    total_mass = 0
    for comp in component_list:
        if comp == "gt":
            mass = P_req / PD_GT
        elif comp == "fc":
            mass = P_req / PD_FC_syst
        elif comp == "hts":
            mass = P_req / PD_HTS
        elif comp == "gt_hex":
            mass = P_req / PD_GT_HEX
        elif comp == "dc_dc":
            mass = P_req / PD_DCDC
        elif comp == "ac_dc":
            mass = P_req / PD_ACDC
        elif comp == "dc_ac":
            mass = P_req / PD_DCAC
        else:
            raise ValueError(f"Unknown component: {comp}")
        total_mass += mass
        return total_mass
    # Add battery mass based on energy requirement


if __name__ == "__main__":
    # Example usage:
    design_a_components = ["gt", "dc_dc", "ac_dc", "dc_ac"]
    mass_a = Design_2_mass(design_a_components, P_bat_req)
    print(f"Design A mass: {mass_a:.2f} kg")