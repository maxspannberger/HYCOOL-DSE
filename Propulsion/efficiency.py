import sys
from pathlib import Path
from pprint import pprint

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from General.component_parameters import component_params as c
from WeightEstimations.mainClassII import ClassIIResult

# Initialise the results from Class II Weight Estimation
ClassIIResults = ClassIIResult()
cable_efficiency = 1

# Obtain mission flight phase duration
t_climb = ClassIIResults.mission.t_climb
t_cruise = ClassIIResults.mission.t_cruise
t_reserve = ClassIIResults.mission.t_reserve

# Obtain power required for flight phases
p_climb = ClassIIResults.power.

# =============================================================================
# Gas Turbine + Battery powertrain
# =============================================================================

# Percentage of electric power in cruise allocated to charging the battery
bat_charge_frac = 0.05


# Efficiency of power directly from gas turbine to motor
gt_eff = (
    c["gt"].efficiency 
    * c["hts"].efficiency 
    * c["ac_dc"].efficiency 
    * c["dc_ac"].efficiency
    * c["hts"].efficiency
    * cable_efficiency
)

# Efficiency of power from gas turbine trough battery to motor
gt_through_bat_eff = (
    c["gt"].efficiency 
    * c["hts"].efficiency 
    * c["ac_dc"].efficiency
    * c["dc_dc"].efficiency  
    * c["bt"].efficiency
    * c["dc_dc"].efficiency
    * c["dc_ac"].efficiency 
    * c["hts"].efficiency
    * cable_efficiency
)

# Efficiency of the total gas turbine 
gt_bat_eff = 

print(gt_eff)