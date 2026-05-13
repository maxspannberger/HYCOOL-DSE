import sys
from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
def return_wanted_params():
    t_climb = get_param('t_climb')             
    t_cruise = get_param('t_cruise')

    P_climb = get_param('P_climb_shaft')       
    P_cruise = get_param('P_cruise_shaft')

    return t_climb, t_cruise, P_climb, P_cruise


# r is P_optimal/P_required
def get_throttle(r):
    # efficiency = max_efficiency * (a*throttle^2 + b*throttle + d)
    a = -0.6
    b = 1.2
    d = 0.4
    throttle = 1/r * (1 - (a + b*r + (d-1)*r**2) / (3*a + 2*b*r + d*r**2))
    eff_factor = a*r**2 + b*r + d
    return throttle, eff_factor


def find_optimal_point(P_opt, P_1, P_2, t_1, t_2):
    r_1 = P_opt/P_1
    r_2 = P_opt/P_2
    throttle_1, eff_1 = get_throttle(r_1)
    throttle_1, eff_2 = get_throttle(r_2)
    goal = P_1 * t_1 * eff_1 + P_2 * t_2 * eff_2
    return goal


def binary_power_search(P_1, P_2, t_1, t_2):
    P_min = min(P_1, P_2)
    P_max = max(P_1, P_2)
    err = (P_max - P_min) / 2
    while err > 1e-6:
        goal_left = find_optimal_point(P_min, P_1, P_2, t_1, t_2)
        goal_right = find_optimal_point(P_max, P_1, P_2, t_1, t_2)
        if goal_left > goal_right:
            P_max = (P_max + P_min) / 2
        else:
            P_min = (P_max + P_min) / 2
        err = (P_max - P_min) / 2
    P_opt = (P_max + P_min) / 2
    return P_opt


# =============================================================================
# Gas Turbine + Battery powertrain
# =============================================================================
def GT_BAT_efficiency(t_charge=1800, cable_efficiency=1.0, show=False):
    t_climb, t_cruise, P_climb, P_cruise = return_wanted_params()

    excess_P_climb = P_climb/P_cruise

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
        * c["dc_dc_2"].efficiency
        * np.sqrt(c["bt"].efficiency)
        * cable_efficiency
    )

    # Efficiency of power from battery to motor (discharge)
    bt_eff_d = (
        np.sqrt(c["bt"].efficiency)
        * c["dc_dc_2"].efficiency
        * c["dc_ac"].efficiency 
        * c["hts_pow"].efficiency
        * cable_efficiency
    )

    bt_c_frac =  (excess_P_climb - 1) / (excess_P_climb + bt_eff_c*bt_eff_d/gt_eff * t_charge/t_climb)

    # climbing efficiency
    climb_eff = excess_P_climb / (1/bt_eff_d + 1/(1-bt_c_frac) * (1/gt_eff - excess_P_climb/bt_eff_d))

    # cruising efficiency
    cruise_eff_c = (1-bt_c_frac)*gt_eff + bt_c_frac*bt_eff_c
    cruise_eff_full = gt_eff

    # component powers
    P_gt = P_cruise / (gt_eff * (1 - bt_c_frac))
    P_bt_discharge = 1/bt_eff_d * (P_climb - P_cruise / (1 - bt_c_frac))
    P_bt_charge = bt_eff_c/gt_eff * bt_c_frac/(1-bt_c_frac) * P_cruise

    # required energies
    E_climb = P_climb * t_climb
    E_cruise_c = (P_cruise + P_bt_charge) * t_charge
    E_cruise_full = P_cruise * (t_cruise - t_charge)

    # total energy efficiency over a flight
    gt_bt_eff = (E_climb * climb_eff + E_cruise_c * cruise_eff_c + E_cruise_full * cruise_eff_full) / (E_climb + E_cruise_c + E_cruise_full)

    if show:
        print("\nGT+BAT")
        print(f"Best charging power fraction: {bt_c_frac}")
        print(f"Climb efficiency: {climb_eff}")
        print(f"Cruise efficiency while charging: {cruise_eff_c}")
        print(f"Cruise efficiency while not charging: {cruise_eff_full}")
        print(f"Total efficiency: {gt_bt_eff}")

    return gt_bt_eff, P_gt, climb_eff, P_bt_discharge, bt_eff_d


# =============================================================================
# Fuel Cell + Battery powertrain
# =============================================================================
def FC_BAT_efficiency(t_charge=1800, cable_efficiency=1.0, show=False):
    t_climb, t_cruise, P_climb, P_cruise = return_wanted_params()

    excess_P_climb = P_climb/P_cruise

    # Efficiency of power from fuel cell to motor
    fc_eff = (
        c["fc_with_hex"].efficiency 
        * c["dc_dc_1"].efficiency
        * c["dc_ac"].efficiency
        * c["hts_pow"].efficiency
        * cable_efficiency
    )

    # Efficiency of power from fuel cell to battery (charge)
    bt_eff_c = (
        c["fc_with_hex"].efficiency 
        * c["dc_dc_1"].efficiency
        * c["dc_dc_2"].efficiency
        * np.sqrt(c["bt"].efficiency)
        * cable_efficiency
    )

    # Efficiency of power from battery to motor (discharge)
    bt_eff_d = (
        np.sqrt(c["bt"].efficiency)
        * c["dc_dc_2"].efficiency
        * c["dc_ac"].efficiency 
        * c["hts_pow"].efficiency
        * cable_efficiency
    )

    bt_c_frac =  (excess_P_climb - 1) / (excess_P_climb + bt_eff_c*bt_eff_d/fc_eff * t_charge/t_climb)

    # climbing efficiency
    climb_eff = excess_P_climb / (1/bt_eff_d + 1/(1-bt_c_frac) * (1/fc_eff - excess_P_climb/bt_eff_d))

    # cruising efficiency
    cruise_eff_c = (1-bt_c_frac)*fc_eff + bt_c_frac*bt_eff_c
    cruise_eff_full = fc_eff

    # component powers
    P_fc = P_cruise / (fc_eff * (1 - bt_c_frac))
    P_bt_discharge = 1/bt_eff_d * (P_climb - P_cruise / (1 - bt_c_frac))
    P_bt_charge = bt_eff_c/fc_eff * bt_c_frac/(1-bt_c_frac) * P_cruise

    # required energies
    E_climb = P_climb * t_climb
    E_cruise_c = (P_cruise + P_bt_charge) * t_charge
    E_cruise_full = P_cruise * (t_cruise - t_charge)

    # total energy efficiency over a flight
    fc_bt_eff = (E_climb * climb_eff + E_cruise_c * cruise_eff_c + E_cruise_full * cruise_eff_full) / (E_climb + E_cruise_c + E_cruise_full)

    if show:
        print("\nFC+BAT")
        print(f"Best charging power fraction: {bt_c_frac}")
        print(f"Climb efficiency: {climb_eff}")
        print(f"Cruise efficiency while charging: {cruise_eff_c}")
        print(f"Cruise efficiency while not charging: {cruise_eff_full}")
        print(f"Total efficiency: {fc_bt_eff}")

    return fc_bt_eff, P_fc, climb_eff, P_bt_discharge, bt_eff_d


# =============================================================================
# Gass Turbine + Gas Turbine powertrain
# =============================================================================
def GT_GT_efficiency(cable_efficiency=1.0, show=False):
    t_climb, t_cruise, P_climb, P_cruise = return_wanted_params()

    excess_P_climb = P_climb/P_cruise

    # Efficiency of power from gas turbine to motor
    gt_eff = (
        c["gt"].efficiency 
        * c["hts_gen"].efficiency 
        * c["ac_dc"].efficiency 
        * c["dc_ac"].efficiency
        * c["hts_pow"].efficiency
        * cable_efficiency
    )

    P_optimal_out = binary_power_search(P_climb, P_cruise, t_climb, t_cruise)
    climb_throttle, climb_eff_factor = get_throttle(P_optimal_out/P_climb)
    cruise_throttle, cruise_eff_factor = get_throttle(P_optimal_out/P_cruise)

    P_optimal_gt = P_optimal_out / (2 * gt_eff)
    P_gt_climb = climb_throttle * P_optimal_gt
    P_gt_cruise = cruise_throttle * P_optimal_gt

    climb_eff = gt_eff * climb_eff_factor
    cruise_eff = gt_eff * cruise_eff_factor

    E_climb = P_climb * t_climb
    E_cruise = P_cruise * t_cruise
    
    # total energy efficiency over a flight
    gt_gt_eff = (E_climb * climb_eff + E_cruise * cruise_eff) / (E_climb + E_cruise)

    if show:
        print("\nGT+GT")
        print(f"Climb throttle: {climb_throttle}")
        print(f"Cruise throttle: {cruise_throttle}")
        print(f"Climb efficiency: {climb_eff}")
        print(f"Cruise efficiency: {cruise_eff}")
        print(f"Total efficiency: {gt_gt_eff}")

    return gt_gt_eff, P_gt_climb, climb_eff, P_gt_cruise, cruise_eff


# =============================================================================
# Gas Turbine + Fuel Cell powertrain
# =============================================================================
def GT_FC_efficiency(cable_efficiency, show):
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

    # Efficiency of power from fuel cell to motor
    fc_eff = (
        c["fc_with_hex"].efficiency 
        * c["dc_dc_1"].efficiency
        * c["dc_ac"].efficiency
        * c["hts_pow"].efficiency
        * cable_efficiency
    )

    # Calculate power of gas turbine and fuel cell, assuming that the gas
    # turbine provides all cruise power and fuel cell provides excess climb power
    P_gt = P_cruise / gt_eff
    P_fc = (P_climb - P_cruise) / fc_eff
    
    # Calculate energy input and outputs for flight phases
    E_out_cruise = P_cruise * t_cruise
    E_out_climb = P_climb * t_climb
    E_in_cruise = P_gt * t_cruise
    E_in_climb = (P_gt + P_fc) * t_climb
    
    # Energy efficiency for cruise and climb
    cruise_eff = E_out_cruise / E_in_cruise
    climb_eff = E_out_climb / E_in_climb
    
    # Total energy efficiency over a flight
    gt_fc_eff = (E_out_climb + E_out_cruise) / (E_in_climb + E_in_cruise)
    
    if show:
        print("\nGT+FC")
        print(f"Climb efficiency: {climb_eff}")
        print(f"Cruise efficiency: {cruise_eff}")
        print(f"Total efficiency: {gt_fc_eff}")

    return gt_fc_eff, P_gt, gt_eff, P_fc, fc_eff



if __name__ == "__main__":
    t_charge = 30*60 # 30 min charge time
    cable_efficiency = 1 # change later

    GT_BAT_efficiency(t_charge=t_charge, cable_efficiency=cable_efficiency, show=True)
    FC_BAT_efficiency(t_charge=t_charge, cable_efficiency=cable_efficiency, show=True)
    GT_GT_efficiency(cable_efficiency=cable_efficiency, show=True)
    GT_FC_efficiency(cable_efficiency=cable_efficiency, show=True)