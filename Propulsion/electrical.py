import numpy as np
import sys
from pathlib import Path
import json

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from General.component_parameters import component_params as comp_params
from WeightEstimations.Aircraft_Config import default_q400_hycool
from WeightEstimations.mainClassII import run_class_ii

T_H2 = 20   # K
T_J_design = 250    # K
T_J_min = 77    # K
R_th_26 = 14.3  # K/kW
R_th = R_th_26/26

cable_loss_per_m = 0.0 # kW/m
P_AC_systems = 310.0  # kW


def get_cable_region_powers(N_motors, N_turbines, component, positions, previous):
    comp_powers = {}
    maximum = {}
    if "in" in component:
        for pos in positions["gt"]:
            comp_powers[pos] = {}
            comp_powers[pos]["length"] = abs(pos - positions["bus"])
            for condition in ["max", "cruise", "OEI_mot", "OEI_gt"]:
                comp_powers[pos][condition] = previous[condition] * (1.0 - cable_loss_per_m * comp_powers[pos]["length"])
                if condition not in maximum:
                    maximum[condition] = 0.0
                if comp_powers[pos][condition] > maximum[condition]:
                    maximum[condition] = comp_powers[pos][condition]

    elif "out" in component:
        for pos in positions["mot"]:
            comp_powers[pos] = {}
            comp_powers[pos]["length"] = abs(pos - positions["bus"])
            for condition in ["max", "cruise", "OEI_mot", "OEI_gt"]:
                comp_powers[pos][condition] = previous[condition] * (1.0 - cable_loss_per_m * comp_powers[pos]["length"])
                if condition == "OEI_mot":
                    maximum[condition] = previous[condition] * (N_motors - 1) / 2
                elif condition == "OEI_gt":
                    maximum[condition] = previous[condition] * N_motors
                else:
                    maximum[condition] = previous[condition] * N_motors / 2

    else:
        raise ValueError("There are only 2 conditions for cabling: before bus (1) and after bus (2).")

    return comp_powers, maximum


def get_powers_per_component(P_max, P_cruise, P_OEI, positions, component_order, comp):
    """
    Returns powers required at every component
    and cable segment powers.
    """

    N_motors = len(positions["mot"]) * 2
    N_turbines = len(positions["gt"]) * 2

    # -------------------------------------------------------------
    # Initial power demand at motor outputs
    # -------------------------------------------------------------
    comp_powers = {}
    comp_powers["out"] = {
        "max": P_max / N_motors,
        "cruise": P_cruise / N_motors,
        "OEI_mot": P_OEI / (N_motors - 1),
        "OEI_gt": P_OEI / N_motors,
    }
    previous = comp_powers["out"].copy()

    # -------------------------------------------------------------
    # Walk upstream through architecture
    # ------------------------------------------------------------

    for component in reversed(component_order):
        comp_powers[component] = {}

        if "cable" in component:
            comp_powers_cable, previous = get_cable_region_powers(N_motors, N_turbines, component, positions, previous)
            comp_powers[component] = comp_powers_cable

        else:
            for condition in previous:
                comp_powers[component][condition] = previous[condition] / comp[component].efficiency
            previous = comp_powers[component].copy()

    return comp_powers


def get_N(P, eff, T_J):
    Q = (1-eff) * P
    N = (T_J - T_H2) / (Q * R_th)
    return int(np.ceil(N))


def get_deltaT(P, eff, N):
    Q = (1-eff) * P
    deltaT  = Q * N * R_th
    return deltaT


def get_P_idle(T_J, N):
    P = (T_J - T_H2) / (R_th * N)
    return P


def size_converter(comp, powers, N=0):
    eff = comp.efficiency
    P_OEI = max(powers["OEI_mot"], powers["OEI_gt"])

    if N == 0:
        N = get_N(powers["max"], eff, T_J_design)
        T_J_max = 250
    else:
        T_J_max = T_H2 + get_deltaT(powers["max"], eff, N)
    T_J_cruise = T_H2 + get_deltaT(powers["cruise"], eff, N)
    T_J_OEI = T_H2 + get_deltaT(P_OEI, eff, N)
    P_heat_idle = get_P_idle(T_J_min, N)

    P_max = max(P_OEI, powers["max"])
    m = P_max / comp.power_density
    max_cooling = P_max * (1 - eff)

    print(f"Number of chips: {N}")
    print(f"Operating temperature: {T_J_max}")
    print(f"Cruise temperature: {T_J_cruise}")
    print(f"OEI temperature: {T_J_OEI}")
    print(f"Idle extra heat: {P_heat_idle}")
    print(f"Max cooling power: {max_cooling}")
    print(f"Mass: {m}")

    return N, T_J_max, T_J_cruise, T_J_OEI, P_heat_idle, m, max_cooling



if __name__ == "__main__":
    # define electrical system architecture
    component_order = ["gt_hex", "hts_gen", "ac_dc", "cable_in", "bus", "cable_out", "dc_ac", "hts_pow"]
    positions = {"gt": [5], "mot": [10, 15], "bus": 3}

    # get class II power results
    print("Performing Class II estimations...")
    cfg = default_q400_hycool()
    class_II_results = run_class_ii(config=3, comp=comp_params, verbose=False, cfg=cfg)
    P_max = class_II_results.P_max_KW
    P_cruise = class_II_results.mission.P_cruise_shaft/1000.0
    P_OEI = class_II_results.weight.P_TO_OEI_KW
    print("Class II estimations finished.")

    # perform power sizing of electrical system
    powers = get_powers_per_component(P_max, P_cruise, P_OEI, positions, component_order, comp=comp_params)
    filename = "power_chain_results.json"
    with open(filename, "w") as f:
        json.dump(powers, f, indent=4)
    print("Electrical system power sizing complete.")

    print("\nINVERTER")
    _, _, _, _, P_inverter, m_inverter, max_cooling_inverter = size_converter(comp_params["dc_ac"], powers["dc_ac"])
    print("\nRECTIFIER")
    _, _, _, _, P_rectifier, m_rectifier, max_cooling_rectifier = size_converter(comp_params["ac_dc"], powers["ac_dc"])
    print("\nBUS")
    _, _, _, _, P_bus, m_bus, max_cooling_bus = size_converter(comp_params["bus"], powers["bus"], N=24)
    
    P_total_idle = 2*len(positions["mot"]) * P_inverter + 2*len(positions["gt"]) * P_rectifier + 2*P_bus
    m_total = 2*len(positions["mot"]) * m_inverter + 2*len(positions["gt"]) * m_rectifier + 2*m_bus
    P_total_idle = 2*len(positions["mot"]) * max_cooling_inverter + 2*len(positions["gt"]) * max_cooling_rectifier + 2*max_cooling_bus

    print(f"\nTotal heating power required for idle: {P_total_idle}")
    print(f"Total mass of converters: {m_total}")
    print("Electrical components sizing complete.")