import sys
from pathlib import Path
from pprint import pprint
import ast
from typing import Optional
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from General.component_parameters import component_params as comp_param

# =============================================================================
# Loading results from Class 2 and calculating the mission phase power and 
# energy requirements.
# =============================================================================
# Create a helper function to pull values safely
def get_param(parameter_name, section_name=None):
    # Load the data
    df = pd.read_csv(root / "outputs/class_ii_results.csv")
    # Clean up whitespace (CSV exports often have hidden spaces in strings)
    df['Section'] = df['Section'].str.strip()
    df['Parameter'] = df['Parameter'].str.strip()
    try:
        if section_name is not None:
            df = df.loc[df['Section'] == section_name]

        # We look for the parameter name and return the associated value
        val = df.loc[df['Parameter'] == parameter_name, 'Value'].values[0]

        if isinstance(val, str):
            val = ast.literal_eval(val)

        if isinstance(val, (tuple, list)):
            val = val[0]

        return float(val)
    except IndexError:
        print(f"Error: Parameter '{parameter_name}' not found in CSV.")
        return None
    except (ValueError, SyntaxError):
        print(f"Error: Parameter '{parameter_name}' in CSV is not a numeric value: {val!r}")
        return None


# Extract your specific variables
def return_wanted_params():
    mission_section = 'Mission Power & Fuel'
    t_climb = get_param('t_climb', mission_section)
    t_cruise = get_param('t_cruise', mission_section)

    P_climb = get_param('P_climb_shaft', mission_section)
    P_cruise = get_param('P_cruise_shaft', mission_section)

    return t_climb, t_cruise, P_climb, P_cruise


# r is P_optimal/P_required
def get_throttle(r):
    # efficiency = max_efficiency * (a*throttle^2 + b*throttle + d)
    a = -0.62659
    b = 1.25318
    d = 0.37341
    
    throttle = 1/r
    eff_factor = a*r**2 + b*r + d
    
    return throttle, eff_factor


def find_optimal_point(P_opt, P_1, P_2, t_1, t_2):
    r_1 = P_opt/P_1
    r_2 = P_opt/P_2
    
    throttle_1, eff_1 = get_throttle(r_1)
    throttle_2, eff_2 = get_throttle(r_2)
    
    eff = (P_1 * t_1 * eff_1 + P_2 * t_2 * eff_2) / (P_1 * t_1 + P_2 * t_2)
    return eff


def golden_power_search(P_1, P_2, t_1, t_2):
    invphi = 2/(1 + np.sqrt(5))
    a = min(P_1, P_2)
    b = max(P_1, P_2)
   
    i = 0
    while b - a > 1e-6 and i < 100:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        if find_optimal_point(c, P_1, P_2, t_1, t_2) > find_optimal_point(d, P_1, P_2, t_1, t_2):
            b = d
        else:  # f(c) > f(d) to find the maximum
            a = c
        i += 1

    return (b + a) / 2


# =============================================================================
# Gas Turbine + Battery powertrain
# =============================================================================
def GT_BAT_efficiency(
    comp: dict,
    t_climb: Optional[float] = None,
    t_cruise: Optional[float] = None,
    P_climb: Optional[float] = None,
    P_cruise: Optional[float] = None,
    t_charge: float = 1800,
    cable_efficiency: float = 1.0,
    show: bool = False,
):

    only_gt_efficiency = comp["gt_hex"].efficiency

    # Efficiency of power from gas turbine to motor
    gt_eff = (
        only_gt_efficiency
        * comp["hts_gen"].efficiency 
        * comp["ac_dc"].efficiency 
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    gt_eff1 = (
        comp["hts_gen"].efficiency 
        * comp["ac_dc"].efficiency 
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    gen_eff = (
        comp["ac_dc"].efficiency 
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    acdc_eff = (
        comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
    )

    dcac_eff = (
        comp["hts_pow"].efficiency
        * cable_efficiency
    )

    dcdc_eff = (
        comp["dc_ac"].efficiency 
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    # Efficiency of power from gas turbine to battery (charge)
    bt_eff_c = (
        comp["gt"].efficiency 
        * comp["hts_gen"].efficiency
        * comp["ac_dc"].efficiency 
        * comp["dc_dc_2"].efficiency
        * np.sqrt(comp["bt"].efficiency)
        * cable_efficiency
    )

    # Efficiency of power from battery to motor (discharge)
    bt_eff_d = (
        np.sqrt(comp["bt"].efficiency)
        * comp["dc_dc_2"].efficiency
        * comp["dc_ac"].efficiency 
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    # iterate to obtain battery charge fraction and optimal power
    error = np.inf
    climb_eff_factor = 1.0
    cruise_eff_factor = 1.0
    bt_c_frac = min(t_climb/t_charge, 0.5)
    P_optimal_gt = P_cruise
    i = 0
    while error > 1 and i < 1000:
        P_optimal_gt_old = P_optimal_gt

        P_gt_climb = P_cruise / (climb_eff_factor * gt_eff * (1 - bt_c_frac))
        P_gt_cruise = P_cruise / (cruise_eff_factor * gt_eff)

        P_optimal_gt = golden_power_search(P_gt_climb, P_gt_cruise, t_climb+t_charge, t_cruise-t_charge)
        climb_throttle, climb_eff_factor = get_throttle(P_optimal_gt/P_gt_climb)
        cruise_throttle, cruise_eff_factor = get_throttle(P_optimal_gt/P_gt_cruise)

        P_bt_discharge = (P_climb - climb_eff_factor * gt_eff * P_gt_climb) / bt_eff_d
        P_bt_charge = t_climb/t_charge * P_bt_discharge

        bt_c_frac = P_bt_charge / (climb_eff_factor * gt_eff * P_gt_climb)
        error = np.abs(P_optimal_gt_old - P_optimal_gt)
        i += 1

    # energy required
    E_climb_out = P_climb * t_climb
    E_cruise_c_out = (P_cruise + P_bt_charge) * t_charge
    E_cruise_full_out = P_cruise * (t_cruise - t_charge)

    # energy provided
    E_climb_in = P_gt_climb * t_climb
    E_cruise_c_in = P_gt_climb * t_charge
    E_cruise_full_in = P_gt_cruise * (t_cruise - t_charge)

    # efficiencies
    climb_eff = E_climb_out / E_climb_in
    cruise_eff_c = E_cruise_c_out / E_cruise_c_in
    cruise_eff_full = E_cruise_full_out / E_cruise_full_in
    cruise_eff = (E_cruise_c_out + E_cruise_full_out) / (E_cruise_c_in + E_cruise_full_in)
    gt_bt_eff = (E_climb_out + E_cruise_c_out + E_cruise_full_out) / (E_climb_in + E_cruise_c_in + E_cruise_full_in)

    if show:
        print("\nGT+BAT")
        print(f"Best charging power fraction: {bt_c_frac}")
        print(f"Climb/cruise & charge throttle: {climb_throttle}")
        print(f"Cruise & not charge throttle: {cruise_throttle}")
        print(f"Climb efficiency: {climb_eff}")
        print(f"Cruise efficiency while charging: {cruise_eff_c}")
        print(f"Cruise efficiency while not charging: {cruise_eff_full}")
        print(f"Cruise average efficiency: {cruise_eff}")
        print(f"Total efficiency: {gt_bt_eff}")

    results_GT_BAT = {
        "LH2-GT-MOT_eff": gt_eff,
        "LH2-GT-BAT_eff": bt_eff_c,
        "GT-MOT_eff": gt_eff1,
        "BAT-MOT_eff": bt_eff_d,
        "GEN_eff": gen_eff,
        "ACDC_eff": acdc_eff,
        "Dcac_eff": dcac_eff,
        "Dcdc_eff": dcdc_eff,
        "Climb_eff": climb_eff,
        "Cruise_charging_eff": cruise_eff_c,
        "Cruise_noncharging_eff": cruise_eff_full,
        "Cruise_average_eff": cruise_eff,
        "Total_eff": gt_bt_eff,
        "GT_P_opt": P_optimal_gt * only_gt_efficiency,
        "GT_throttle_climb": climb_throttle,
        "GT_throttle_cruise": cruise_throttle,
        "BAT_P_discharge": P_bt_discharge,
        "BAT_charging_frac": bt_c_frac
    }

    return results_GT_BAT


# =============================================================================
# Fuel Cell + Battery powertrain
# =============================================================================
def FC_BAT_efficiency(
    comp: dict,
    t_climb: Optional[float] = None,
    t_cruise: Optional[float] = None,
    P_climb: Optional[float] = None,
    P_cruise: Optional[float] = None,
    t_charge=1800,
    cable_efficiency=1.0,
    show=False,
):

    only_fc_efficiency = comp["fc_with_hex"].efficiency

    # Efficiency of power from fuel cell to motor
    fc_eff = (
        comp["fc_with_hex"].efficiency 
        * comp["dc_dc_1"].efficiency
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    fc_eff1 = (
        comp["dc_dc_1"].efficiency
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    dcdc_1_eff = (
        comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    # Efficiency of power from fuel cell to battery (charge)
    bt_eff_c = (
        comp["fc_with_hex"].efficiency 
        * comp["dc_dc_1"].efficiency
        * comp["dc_dc_2"].efficiency
        * np.sqrt(comp["bt"].efficiency)
        * cable_efficiency
    )

    # Efficiency of power from battery to motor (discharge)
    bt_eff_d = (
        np.sqrt(comp["bt"].efficiency)
        * comp["dc_dc_2"].efficiency
        * comp["dc_ac"].efficiency 
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    # iterate to obtain battery charge fraction and optimal power
    error = np.inf
    bt_c_frac = min(t_climb/t_charge, 0.5)
    i = 0
    while error > 1e-8 and i < 1000:
        bt_c_frac_old = bt_c_frac

        P_fc_climb = P_cruise / (fc_eff * (1 - bt_c_frac))
        P_fc_cruise = P_cruise / fc_eff

        P_bt_discharge = (P_climb - fc_eff * P_fc_climb) / bt_eff_d
        P_bt_charge = t_climb/t_charge * P_bt_discharge

        bt_c_frac = P_bt_charge / (fc_eff * P_fc_climb)
        error = np.abs(bt_c_frac_old - bt_c_frac)
        i += 1

    # energy required
    E_climb_out = P_climb * t_climb
    E_cruise_c_out = (P_cruise + P_bt_charge) * t_charge
    E_cruise_full_out = P_cruise * (t_cruise - t_charge)

    # energy provided
    E_climb_in = P_fc_climb * t_climb
    E_cruise_c_in = P_fc_climb * t_charge
    E_cruise_full_in = P_fc_cruise * (t_cruise - t_charge)

    # efficiencies
    climb_eff = E_climb_out / E_climb_in
    cruise_eff_c = E_cruise_c_out / E_cruise_c_in
    cruise_eff_full = E_cruise_full_out / E_cruise_full_in
    cruise_eff = (E_cruise_c_out + E_cruise_full_out) / (E_cruise_c_in + E_cruise_full_in)
    fc_bt_eff = (E_climb_out + E_cruise_c_out + E_cruise_full_out) / (E_climb_in + E_cruise_c_in + E_cruise_full_in)

    if show:
        print("\nFC+BAT")
        print(f"Best charging power fraction: {bt_c_frac}")
        print(f"Climb efficiency: {climb_eff}")
        print(f"Cruise efficiency while charging: {cruise_eff_c}")
        print(f"Cruise efficiency while not charging: {cruise_eff_full}")
        print(f"Cruise average efficiency: {cruise_eff}")
        print(f"Total efficiency: {fc_bt_eff}")

    results_FC_BAT = {
        "LH2-FC-MOT_eff": fc_eff,
        "LH2-FC-BAT_eff": bt_eff_c,
        "FC-MOT_eff": fc_eff1,
        "DC-DC1_eff": dcdc_1_eff,
        "BAT-MOT_eff": bt_eff_d,
        "Climb_eff": climb_eff,
        "Cruise_charging_eff": cruise_eff_c,
        "Cruise_noncharging_eff": cruise_eff_full,
        "Cruise_average_eff": cruise_eff,
        "Total_eff": fc_bt_eff,
        "FC_P": 0.5 * P_fc_climb * only_fc_efficiency,
        # "FC_P_cruise": 0.5 * P_fc_cruise * only_fc_efficiency, # not used for sizing, might be useful if we vary FC efficiency with throttle
        "BAT_P_discharge": P_bt_discharge,
        "BAT_charging_frac": bt_c_frac
    }

    return results_FC_BAT


# =============================================================================
# Gass Turbine + Gas Turbine powertrain
# =============================================================================
def GT_GT_efficiency(
    comp: dict,
    t_climb: Optional[float] = None,
    t_cruise: Optional[float] = None,
    P_climb: Optional[float] = None,
    P_cruise: Optional[float] = None,
    cable_efficiency=1.0,
    show=False,
):

    only_gt_efficiency = comp["gt_hex"].efficiency

    # Efficiency of power from gas turbine to motor
    gt_eff = (
        only_gt_efficiency
        * comp["hts_gen"].efficiency 
        * comp["ac_dc"].efficiency 
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    gt_eff1 = (
        comp["hts_gen"].efficiency 
        * comp["ac_dc"].efficiency 
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    P_optimal_out = golden_power_search(P_climb, P_cruise, t_climb, t_cruise)
    climb_throttle, climb_eff_factor = get_throttle(P_optimal_out/P_climb)
    cruise_throttle, cruise_eff_factor = get_throttle(P_optimal_out/P_cruise)

    P_optimal_gt = P_optimal_out / gt_eff
    P_gt_climb = climb_throttle * P_optimal_gt
    P_gt_cruise = cruise_throttle * P_optimal_gt

    # energy required
    E_climb_out = P_climb * t_climb
    E_cruise_out = P_cruise * t_cruise

    # energy provided
    E_climb_in = P_gt_climb * t_climb / climb_eff_factor
    E_cruise_in = P_gt_cruise * t_cruise / cruise_eff_factor
    
    # efficiencies
    climb_eff = E_climb_out/E_climb_in
    cruise_eff = E_cruise_out/E_cruise_in

    gt_gt_eff = (E_climb_out + E_cruise_out) / (E_climb_in + E_cruise_in)

    if show:
        print("\nGT+GT")
        print(f"Climb throttle: {climb_throttle}")
        print(f"Cruise throttle: {cruise_throttle}")
        print(f"Climb efficiency: {climb_eff}")
        print(f"Cruise efficiency: {cruise_eff}")
        print(f"Total efficiency: {gt_gt_eff}")

    results_GT_GT = {
        "LH2-GT-MOT_eff": gt_eff,
        "GT-MOT_eff": gt_eff1,
        "Climb_eff": climb_eff,
        "Cruise_average_eff": cruise_eff,
        "Total_eff": gt_gt_eff,
        "GT_P_opt": 0.5 * P_optimal_gt * only_gt_efficiency,
        "GT_throttle_climb": climb_throttle,
        "GT_throttle_cruise": cruise_throttle
    }

    return results_GT_GT


# =============================================================================
# Gas Turbine + Fuel Cell powertrain
# =============================================================================
def GT_FC_efficiency(
    comp: dict,
    t_climb: Optional[float] = None,
    t_cruise: Optional[float] = None,
    P_climb: Optional[float] = None,
    P_cruise: Optional[float] = None,
    P_OEI_out=2.6e6, 
    cable_efficiency=1.0,
    show=False,
):

    only_gt_efficiency = comp["gt_hex"].efficiency
    only_fc_efficiency = comp["fc_with_hex"].efficiency

    # Efficiency of power from gas turbine to motor
    gt_eff = (
        only_gt_efficiency
        * comp["hts_gen"].efficiency 
        * comp["ac_dc"].efficiency 
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    #Efficiency of power after gas turbine to motor
    gt1_eff = (
        comp["hts_gen"].efficiency 
        * comp["ac_dc"].efficiency 
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    # Efficiency of power from fuel cell to motor
    fc_eff = (
        only_fc_efficiency 
        * comp["dc_dc_1"].efficiency
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    # Efficiency of power from fuel cell to motor
    fc_eff1 = (
        comp["dc_dc_1"].efficiency
        * comp["dc_ac"].efficiency
        * comp["hts_pow"].efficiency
        * cable_efficiency
    )

    P_fc = P_OEI_out / fc_eff
    P_climb_by_gt = P_climb - P_OEI_out
    P_cruise_by_gt = P_cruise - P_OEI_out
    P_optimal_out_gt = golden_power_search(P_climb_by_gt, P_cruise_by_gt, t_climb, t_cruise)

    climb_throttle, climb_eff_factor = get_throttle(P_optimal_out_gt/P_climb_by_gt)
    cruise_throttle, cruise_eff_factor = get_throttle(P_optimal_out_gt/P_cruise_by_gt)

    P_optimal_gt = P_optimal_out_gt / gt_eff
    P_gt_climb = climb_throttle * P_optimal_gt
    P_gt_cruise = cruise_throttle * P_optimal_gt

    # energy required
    E_cruise_out = P_cruise * t_cruise
    E_climb_out = P_climb * t_climb

    # energy provided
    E_cruise_in = (P_gt_cruise + P_fc) * t_cruise
    E_climb_in = (P_gt_climb + P_fc) * t_climb 

    # efficiencies
    cruise_eff = E_cruise_out / E_cruise_in
    climb_eff = E_climb_out / E_climb_in
    gt_fc_eff = (E_climb_out + E_cruise_out) / (E_climb_in + E_cruise_in)
    
    if show:
        print("\nGT+FC")
        print(f"Climb efficiency: {climb_eff}")
        print(f"Cruise efficiency: {cruise_eff}")
        print(f"Total efficiency: {gt_fc_eff}")


    results_GT_FC = {
        "LH2-GT-MOT_eff": gt_eff,
        "LH2-FC-MOT_eff": fc_eff,
        "GT-MOT_eff": gt1_eff,
        "FC-MOT_eff": fc_eff1,
        "Climb_eff": climb_eff,
        "Cruise_average_eff": cruise_eff,
        "Total_eff": gt_fc_eff,
        "FC_P": 0.5 * P_fc * only_fc_efficiency,
        "GT_P_opt": 0.5 * P_optimal_gt * only_gt_efficiency,
        "GT_throttle_climb": climb_throttle,
        "GT_throttle_cruise": cruise_throttle
    }

    return results_GT_FC



if __name__ == "__main__":
    t_charge = 30*60 # 30 min charge time
    cable_efficiency = 1 # change later
    t_climb, t_cruise, P_climb, P_cruise = return_wanted_params()

    # results_GT_BAT = GT_BAT_efficiency(comp=comp_param, t_charge=t_charge, cable_efficiency=cable_efficiency, show=True,t_climb=t_climb, t_cruise=t_cruise, P_climb=P_climb, P_cruise=P_cruise)
    # # print(results_GT_BAT)

    # results_FC_BAT = FC_BAT_efficiency(comp=comp_param, t_charge=t_charge, cable_efficiency=cable_efficiency, show=True,t_climb=t_climb, t_cruise=t_cruise, P_climb=P_climb, P_cruise=P_cruise)
    # # #print(results_FC_BAT)

    results_GT_GT = GT_GT_efficiency(comp=comp_param, cable_efficiency=cable_efficiency, show=True,t_climb=t_climb, t_cruise=t_cruise, P_climb=P_climb, P_cruise=P_cruise)
    # #print(results_GT_GT)

    results_GT_FC = GT_FC_efficiency(comp=comp_param, cable_efficiency=cable_efficiency, show=True,t_climb=t_climb, t_cruise=t_cruise, P_climb=P_climb, P_cruise=P_cruise)
    # # print(results_GT_FC)