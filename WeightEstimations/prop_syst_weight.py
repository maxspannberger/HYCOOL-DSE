import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
root = Path(_file_).resolve().parent.parent
sys.path.append(str(root))
from General.component_parameters import component_params as c


#--------------------- INPUTS ------------------------- (these will be changed to take proper variables from other files)
P_cruise    =   3.8  #MW 
P_climb     =   5.1  #MW
P_reserve   =   1.3  #MW
P_TO_OEI    =   3.09 #MW

N_propulsors = 2

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
