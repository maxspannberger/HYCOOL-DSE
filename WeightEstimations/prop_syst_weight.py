import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from General.component_parameters import component_params as c
from mainClassII import default_q400_hycool, run_class_ii

# Run Class II once and reuse the config/result
cfg = default_q400_hycool()
class_ii_result = run_class_ii(cfg, verbose=False)

#--------------------- INPUTS ------------------------- (these will be changed to take proper variables from other files)

P_cruise    =   class_ii_result.mission.P_cruise_shaft/ 1e6  #MW 
P_climb     =   class_ii_result.mission.P_climb_shaft/ 1e6  #MW
P_reserve   =   class_ii_result.mission.P_reserve_shaft/ 1e6  #MW

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
