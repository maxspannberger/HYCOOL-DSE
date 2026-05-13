import sys
from pathlib import Path
from pprint import pprint
import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from General.component_parameters import component_params as c

# =============================================================================
# Loading results from Class 2 and calculating the mission phase power and 
# energy requirements.
# =============================================================================

# Load the data
df = pd.read_csv(root / "outputs/class_ii_results.csv")
# Clean up whitespace (CSV exports often have hidden spaces in strings)
df['Section'] = df['Section'].str.strip()
df['Parameter'] = df['Parameter'].str.strip()

# Create a helper function to pull values safely
def get_param(parameter_name):
    try:
        # We look for the parameter name and return the associated value
        val = df.loc[df['Parameter'] == parameter_name, 'Value'].values[0]
        return float(val)
    except IndexError:
        print(f"Error: Parameter '{parameter_name}' not found in CSV.")
        return None

# Extract your specific variables
t_climb = get_param('t_climb')             
t_cruise = get_param('t_cruise')           
t_reserve = get_param('t_reserve')         

P_climb = get_param('P_climb_shaft')       
P_cruise = get_param('P_cruise_shaft')

# Energy per flight phase that has to arrive at the shaft
E_climb = P_climb * t_climb
E_cruise = P_cruise * t_cruise
E_total = E_climb + E_cruise

cable_efficiency = 1

# =============================================================================
# Gas Turbine + Battery powertrain
# =============================================================================

# Efficiency of power directly from gas turbine to motor
gt_eff = (
    c["gt"].efficiency 
    * c["hts_gen"].efficiency 
    * c["ac_dc"].efficiency 
    * c["dc_ac"].efficiency
    * c["hts_pow"].efficiency
    * cable_efficiency
)

# Efficiency of power from gas turbine trough battery to motor
bt_eff = (
    c["bt"].efficiency
    * c["dc_dc"].efficiency
    * c["dc_ac"].efficiency 
    * c["hts_pow"].efficiency
    * cable_efficiency
)

# Fraction of shaft power in climb that is provided by the battery and gas turbine
P_frac_bt_cruise = (P_climb - P_cruise) / (P_climb)
P_frac_gt_cruise = 1 - P_frac_bt_cruise

# Efficiency of the total gas turbine 
gt_bt_eff = ((E_cruise / E_total) * gt_eff + (E_climb / E_total) / 
             (P_frac_gt_cruise / gt_eff + P_frac_bt_cruise / bt_eff))

print(gt_bt_eff)