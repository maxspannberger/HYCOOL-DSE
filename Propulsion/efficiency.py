import sys
from pathlib import Path
from pprint import pprint
import pandas as pd
<<<<<<< HEAD
=======
import numpy as np
from matplotlib import pyplot as plt
>>>>>>> de8da238cd18af0e61728c3100155996387a0f39

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from General.component_parameters import component_params as c

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
def return_wanted_params():
    t_climb = get_param('t_climb')             
    t_cruise = get_param('t_cruise')

    P_climb = get_param('P_climb_shaft')       
    P_cruise = get_param('P_cruise_shaft')

    return t_climb, t_cruise, P_climb, P_cruise



# =============================================================================
# Gas Turbine + Battery powertrain
# =============================================================================
def GT_BAT_efficiency(bt_c_frac=0.05, cable_efficiency=1.0):
    t_climb, t_cruise, P_climb, P_cruise = return_wanted_params()

    # Efficiency of power from gas turbine to motor
    gt_eff = (
        c["gt"].efficiency 
        * c["hts_gen"].efficiency 
        * c["ac_dc"].efficiency 
        * c["dc_ac"].efficiency
        * c["hts_pow"].efficiency
        * cable_efficiency
    )

    # Efficiency of power from gas turbine to battery (charge)
    bt_eff_c = (
        c["gt"].efficiency 
        * c["ac_dc"].efficiency 
        * c["dc_dc"].efficiency
        # * c["bt"].efficiency
        * cable_efficiency
    )

    # Efficiency of power from battery to motor (discharge)
    bt_eff_d = (
        c["bt"].efficiency
        * c["dc_dc"].efficiency
        * c["dc_ac"].efficiency 
        * c["hts_pow"].efficiency
        * cable_efficiency
    )

    # how much more power we have for climb than for cruise
    excess_P_climb = P_climb/P_cruise

    # climbing efficiency
    climb_eff = excess_P_climb / (1/bt_eff_d + 1/(1-bt_c_frac) * (1/gt_eff - excess_P_climb/bt_eff_d))

    # cruising efficiency
    cruise_eff = (1-bt_c_frac)*gt_eff + bt_c_frac*bt_eff_c

    # total energy efficiency over a flight
    gt_bt_eff = (t_climb * climb_eff + t_cruise * cruise_eff) / (t_climb + t_cruise)

    # print(f"Climb efficiency: {climb_eff}")
    # print(f"Cruise efficiency: {cruise_eff}")
    # print(f"Total efficiency: {gt_bt_eff}")

    return gt_bt_eff



if __name__ == "__main__":
    # bt_c_frac = 0.05 # how much power goes to charging the battery while cruising
    cable_efficiency = 1 # change later

    gt_bt_eff = []
    bt_c_frac_range = np.linspace(0.00, 0.99, 100)
    for bt_c_frac in bt_c_frac_range:
        gt_bt_eff.append(GT_BAT_efficiency(bt_c_frac=bt_c_frac, cable_efficiency=cable_efficiency))
    
    plt.plot(bt_c_frac_range, gt_bt_eff)
    plt.show()