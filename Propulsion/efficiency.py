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
# Create a helper function to pull values safely
def get_param(parameter_name):
    # Load the data
    df = pd.read_csv(root / "outputs/class_ii_results.csv")
    # Clean up whitespace (CSV exports often have hidden spaces in strings)
    df['Section'] = df['Section'].str.strip()
    df['Parameter'] = df['Parameter'].str.strip()
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
    throttle_2, eff_2 = get_throttle(r_2)
    eff = (P_1 * t_1 * eff_1 + P_2 * t_2 * eff_2) / (P_1 * t_1 + P_2 * t_2)
    return eff


def binary_power_search(P_1, P_2, t_1, t_2):
    P_min = min(P_1, P_2)
    P_max = max(P_1, P_2)
    err = (P_max - P_min) / 2
    while err > 1e-6:
        eff_left = find_optimal_point(P_min, P_1, P_2, t_1, t_2)
        eff_right = find_optimal_point(P_max, P_1, P_2, t_1, t_2)
        if eff_left > eff_right:
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

    only_gt_efficiency = c["gt"].efficiency

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

    # iterate to obtain battery charge fraction and optimal power
    error = np.inf
    climb_eff_factor = 1.0
    bt_c_frac = 0.0
    while error > 1e-8:
        bt_c_frac_old = bt_c_frac
        bt_c_frac = (excess_P_climb - 1) / (excess_P_climb + bt_eff_c*bt_eff_d/(gt_eff*climb_eff_factor) * t_charge/t_climb)
        error = np.abs(bt_c_frac_old - bt_c_frac)

        P_optimal_out = binary_power_search((1-bt_c_frac)*P_climb, P_cruise, t_climb+t_charge, t_cruise-t_charge)
        climb_throttle, climb_eff_factor = get_throttle(P_optimal_out/((1-bt_c_frac)*P_climb))
        cruise_throttle, cruise_eff_factor = get_throttle(P_optimal_out/P_cruise)

    # component powers
    P_bt_discharge = 1/bt_eff_d * (P_climb - P_cruise / (1 - bt_c_frac))
    P_bt_charge = bt_eff_c/gt_eff * bt_c_frac/(1-bt_c_frac) * P_cruise
    P_optimal_gt = P_optimal_out / (gt_eff * (1 - bt_c_frac)) * only_gt_efficiency
    P_gt_climb = climb_throttle * P_optimal_gt
    P_gt_cruise = cruise_throttle * P_optimal_gt

    # climbing efficiency
    climb_eff = excess_P_climb / (1/bt_eff_d + 1/(1-bt_c_frac) * (1/(gt_eff*climb_eff_factor) - excess_P_climb/bt_eff_d))

    # cruising efficiency
    cruise_eff_c = (1-bt_c_frac)*gt_eff*climb_eff_factor + bt_c_frac*bt_eff_c
    cruise_eff_full = gt_eff*cruise_eff_factor

    # required energies
    E_climb = P_climb * t_climb
    E_cruise_c = (P_cruise + P_bt_charge) * t_charge
    E_cruise_full = P_cruise * (t_cruise - t_charge)

    # total energy efficiency over a flight
    gt_bt_eff = (E_climb * climb_eff + E_cruise_c * cruise_eff_c + E_cruise_full * cruise_eff_full) / (E_climb + E_cruise_c + E_cruise_full)

    if show:
        print("\nGT+BAT")
        print(f"Best charging power fraction: {bt_c_frac}")
        print(f"Climb/cruise & charge throttle: {climb_throttle}")
        print(f"Cruise & not charge throttle: {cruise_throttle}")
        print(f"Climb efficiency: {climb_eff}")
        print(f"Cruise efficiency while charging: {cruise_eff_c}")
        print(f"Cruise efficiency while not charging: {cruise_eff_full}")
        print(f"Total efficiency: {gt_bt_eff}")

    results_GT_BAT = {
        "LH2-GT-MOT_eff": gt_eff,
        "LH2-GT-BAT_eff": bt_eff_c,
        "BAT-MOT_eff": bt_eff_d,
        "Climb_eff": climb_eff,
        "Cruise_charging_eff": cruise_eff_c,
        "Cruise_noncharging_eff": cruise_eff_full,
        "Total_eff": gt_bt_eff,
        "GT_P_opt": P_optimal_gt,
        "GT_throttle_climb": climb_throttle,
        "GT_throttle_cruise": cruise_throttle,
        "BAT_P_discharge": P_bt_discharge,
        "BAT_charging_frac": bt_c_frac
    }

    return results_GT_BAT


# =============================================================================
# Fuel Cell + Battery powertrain
# =============================================================================
def FC_BAT_efficiency(t_charge=1800, cable_efficiency=1.0, show=False):
    t_climb, t_cruise, P_climb, P_cruise = return_wanted_params()

    excess_P_climb = P_climb/P_cruise

    only_fc_efficiency = c["fc_with_hex"].efficiency

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
    P_fc = P_cruise / (fc_eff * (1 - bt_c_frac)) * only_fc_efficiency
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

    results_FC_BAT = {
        "LH2-FC-MOT_eff": fc_eff,
        "LH2-FC-BAT_eff": bt_eff_c,
        "BAT-MOT_eff": bt_eff_d,
        "Climb_eff": climb_eff,
        "Cruise_charging_eff": cruise_eff_c,
        "Cruise_noncharging_eff": cruise_eff_full,
        "Total_eff": fc_bt_eff,
        "FC_P": P_fc,
        "BAT_P_discharge": P_bt_discharge,
        "BAT_charging_frac": bt_c_frac
    }

    return results_FC_BAT


# =============================================================================
# Gass Turbine + Gas Turbine powertrain
# =============================================================================
def GT_GT_efficiency(cable_efficiency=1.0, show=False):
    t_climb, t_cruise, P_climb, P_cruise = return_wanted_params()

    excess_P_climb = P_climb/P_cruise

    only_gt_efficiency = c["gt"].efficiency

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

    P_optimal_gt = P_optimal_out / (2 * gt_eff) * only_gt_efficiency
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

    results_GT_GT = {
        "LH2-GT-MOT_eff": gt_eff,
        "Climb_eff": climb_eff,
        "Cruise_eff": cruise_eff,
        "Total_eff": gt_gt_eff,
        "GT_P_opt": P_optimal_gt,
        "GT_throttle_climb": climb_throttle,
        "GT_throttle_cruise": cruise_throttle
    }

    return results_GT_GT


# =============================================================================
# Gas Turbine + Fuel Cell powertrain
# =============================================================================
def GT_FC_efficiency(cable_efficiency=1.0, show=False):
    t_climb, t_cruise, P_climb, P_cruise = return_wanted_params()

    only_gt_efficiency = c["gt"].efficiency
    only_fc_efficiency = c["fc_with_hex"].efficiency

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
    P_gt = P_cruise / gt_eff * only_gt_efficiency
    P_fc = (P_climb - P_cruise) / fc_eff * only_fc_efficiency
    
    # Calculate energy input and outputs for flight phases
    E_out_cruise = P_cruise * t_cruise
    E_out_climb = P_climb * t_climb
    E_in_cruise = P_gt * t_cruise / only_gt_efficiency
    E_in_climb = (P_gt/only_gt_efficiency + P_fc/only_fc_efficiency) * t_climb 
    
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

    results_GT_FC = {
        "LH2-GT-MOT_eff": gt_eff,
        "LH2-FC-MOT_eff": fc_eff,
        "Climb_eff": climb_eff,
        "Cruise_eff": cruise_eff,
        "Total_eff": gt_fc_eff,
        "GT_P": P_gt,
        "FC_P": P_fc
    }

    return results_GT_FC



if __name__ == "__main__":
    t_charge = 30*60 # 30 min charge time
    cable_efficiency = 1 # change later

    results_GT_BAT = GT_BAT_efficiency(t_charge=t_charge, cable_efficiency=cable_efficiency, show=True)
    # print(results_GT_BAT)

    results_FC_BAT = FC_BAT_efficiency(t_charge=t_charge, cable_efficiency=cable_efficiency, show=True)
    # print(results_FC_BAT)

    results_GT_GT = GT_GT_efficiency(cable_efficiency=cable_efficiency, show=True)
    # print(results_GT_GT)

    results_GT_FC = GT_FC_efficiency(cable_efficiency=cable_efficiency, show=True)
    # print(results_GT_FC)