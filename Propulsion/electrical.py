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

V = 3000 # V
rho_c = 6380 # kg/m^3
rho_i = 900 # kg/m^3
J_0 = 2.5e9 # A/m^2
dJdB = -0.07 # exponent
U_i = 5e7 # V/m
mu_r = 2.6 # -
mu_0 = np.pi*4e-7 # T*m/A


def get_cable_region_powers(component, positions, previous):
    comp_powers = {}
    maximum = {}
    maximum[positions["gt"][0]] = {}

    if "in" in component:
        positions_gt = positions["gt"] + [-x for x in positions["gt"]]
        for pos in positions_gt:
            comp_powers[pos] = {}
            comp_powers[pos]["length"] = abs(pos - positions["bus"])

            for condition in ["max", "cruise", "OEI_mot", "OEI_gt", "OEI_bus"]:
                if condition == "OEI_gt":
                    comp_powers[pos][condition] = previous[abs(pos)][condition] * (1.0 - cable_loss_per_m * comp_powers[pos]["length"])
                elif condition == "OEI_bus":
                    comp_powers[pos][condition] = 0.5 * previous[abs(pos)][condition] * (1.0 - cable_loss_per_m * comp_powers[pos]["length"])
                else:
                    if pos < 0:
                        comp_powers[pos][condition] = 0.0
                    else:
                        comp_powers[pos][condition] = previous[abs(pos)][condition] * (1.0 - cable_loss_per_m * comp_powers[pos]["length"])
                
                if condition not in maximum[positions["gt"][0]]:
                    maximum[positions["gt"][0]][condition] = 0.0
                if comp_powers[pos][condition] > maximum[positions["gt"][0]][condition]:
                    maximum[positions["gt"][0]][condition] = comp_powers[pos][condition]

    elif "out" in component:
        positions_mot = positions["mot"] + [-x for x in positions["mot"]]
        fracs = positions["mot_frac"] + positions["mot_frac"]

        for pos, frac in zip(positions_mot, fracs):
            comp_powers[pos] = {}
            comp_powers[pos]["length"] = abs(pos - positions["bus"])

            for condition in ["max", "cruise", "OEI_mot", "OEI_gt", "OEI_bus"]:                
                if condition == "OEI_mot":
                    comp_powers[pos][condition] = 0.5 * previous[abs(pos)][condition] * (1.0 - cable_loss_per_m * comp_powers[pos]["length"])
                elif condition == "OEI_bus":
                    comp_powers[pos][condition] = previous[abs(pos)][condition] * (1.0 - cable_loss_per_m * comp_powers[pos]["length"])
                else:
                    if pos < 0:
                        comp_powers[pos][condition] = 0.0
                    else:
                            comp_powers[pos][condition] = previous[abs(pos)][condition] * (1.0 - cable_loss_per_m * comp_powers[pos]["length"])

                if condition not in maximum[positions["gt"][0]]:
                    maximum[positions["gt"][0]][condition] = 0.0
                if condition == "OEI_mot":
                    maximum[positions["gt"][0]][condition] = max(previous[abs(pos)][condition] / frac * (1 + min(fracs)) / 2, maximum[positions["gt"][0]][condition])
                elif condition == "OEI_bus":
                    maximum[positions["gt"][0]][condition] = max(previous[abs(pos)][condition] / frac * 2, maximum[positions["gt"][0]][condition])
                else:
                    maximum[positions["gt"][0]][condition] = max(previous[abs(pos)][condition] / frac, maximum[positions["gt"][0]][condition])

    else:
        raise ValueError("There are only 2 conditions for cabling: before bus (1) and after bus (2).")

    return comp_powers, maximum


def get_powers_per_component(P_max, P_cruise, P_OEI, positions, component_order, comp):
    """
    Returns powers required at every component
    and cable segment powers.
    """

    # -------------------------------------------------------------
    # Initial power demand at motor outputs
    # -------------------------------------------------------------
    comp_powers = {}
    comp_powers["out"] = {}
    for pos, frac in zip(positions["mot"], positions["mot_frac"]):
        comp_powers["out"][pos] = {
                "max": P_max / 2 * frac,
                "cruise": P_cruise / 2 * frac,
                "OEI_mot": P_OEI * (1.0 - 0.5 * max(positions["mot_frac"])) / 2 * frac,
                "OEI_gt": P_OEI / 2 * frac,
                "OEI_bus": P_OEI  * frac / 2
            }

    previous = comp_powers["out"].copy()

    # -------------------------------------------------------------
    # Walk upstream through architecture
    # ------------------------------------------------------------

    for component in reversed(component_order):
        comp_powers[component] = {}
        print(component)

        if "cable" in component:
            comp_powers_cable, previous = get_cable_region_powers(component, positions, previous)
            comp_powers[component] = comp_powers_cable

        else:
            for pos in previous:
                comp_powers[component][pos] = {}
                for condition in previous[pos]:
                    comp_powers[component][pos][condition] = previous[pos][condition] / comp[component].efficiency
            previous = comp_powers[component].copy()

    filename = "power_chain_results.json"
    with open(filename, "w") as f:
        json.dump(comp_powers, f, indent=4)

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


def size_all_converters(compontents, comp=comp_params):
    converter_sizing = {}
    P_heat_total = 0.0
    P_cool_total = 0.0
    m_total = 0.0

    for component in compontents:
        converter_sizing[component] = {}
        if component == "bus":
            N = 24
        else:
            N = 0

        for pos in powers[component]:
            converter_sizing[component][pos] = {}
            _, _, _, _, P_heat, mass, P_cool = size_converter(comp[component], powers[component][pos], N=N)
            converter_sizing[component][pos]["P_heat"] = P_heat
            converter_sizing[component][pos]["P_cool"] = P_cool
            converter_sizing[component][pos]["mass"] = mass

            P_heat_total += P_heat * 2
            P_cool_total += P_cool * 2
            m_total += mass * 2
        
    converter_sizing["total"] = {
        "P_heat": P_heat_total,
        "P_cool": P_cool_total,
        "mass": m_total
    }

    filename = "converter_sizing_results.json"
    with open(filename, "w") as f:
        json.dump(converter_sizing, f, indent=4)

    return converter_sizing


def get_maximum_powers(powers):
    max_powers = {}
    length = 0.0

    for component in powers:
        for pos in powers[component]:
            if "cable" in component:
                if "cable" not in max_powers:
                    max_powers["cable"] = 0.0
                for condition in powers[component][pos]:
                    if condition != "length":
                        if powers[component][pos][condition] > max_powers["cable"]:
                            max_powers["cable"] = powers[component][pos][condition]
                    else:
                        length += powers[component][pos][condition]
            else:
                if component not in max_powers:
                    max_powers[component] = {}
                if pos not in max_powers[component]:
                    max_powers[component][pos] = 0.0
                for condition in powers[component][pos]:
                    if powers[component][pos][condition] > max_powers[component][pos]:
                        max_powers[component][pos] = powers[component][pos][condition]

    filename = "max_power_results.json"
    with open(filename, "w") as f:
        json.dump(max_powers, f, indent=4)

    length *= 2 # only counted cables connected to one bus

    return max_powers, length


def size_cables(max_powers, length=200, N_cables=6, SF=1):
    P = max_powers["cable"] * 1000 # W
    V_max = V*SF
    I = P/V
    I_max = I*SF

    iteration_finished = False
    d_old = np.inf
    i = 0
    while not iteration_finished and i < 100:
        B = (N_cables * mu_r * mu_0 * I_max) / (2 * np.pi * d_old)
        J = J_0 * 10 ** (dJdB * B)
        A_c = I_max / J
        r_c = np.sqrt(A_c / np.pi)
        t_i = V_max / U_i
        d_new = 2 * (r_c + t_i)

        if abs(d_old - d_new) < 1e-10:
            iteration_finished = True
        i += 1
        d_old = d_new
    
    print(f"\nOptimal wire radius: {r_c}")
    print(f"Optimal insulator thickness: {t_i}")

    t_i = round(0.0001 * np.ceil(10000 * t_i), 4)
    r_c = round(0.00005 * np.ceil(20000 * r_c), 4)
    d = round(2 * (r_c + t_i), 4)
    A_c = np.pi * r_c**2
    A_i = np.pi * t_i * (2*r_c + t_i)
    mass_density = A_c * rho_c + A_i * rho_i
    mass = mass_density * length
    print(f"\nConservative wire radius [mm]: {1000*r_c}")
    print(f"Conservative insulator thickness [mm]: {1000*t_i}")
    print(f"Wire diameter [mm]: {1000*d}")
    print(f"Cable mass density [kg/m]: {mass_density:.6f}")
    print(f"Cable mass [kg]: {mass:.3f}")

    results = {
        "r_core": r_c,
        "t_insulation": t_i,
        "d_cable": d,
        "m_density": mass_density,
        "m": mass
    }

    return results



if __name__ == "__main__":
    # define electrical system architecture
    component_order = ["gt_hex", "hts_gen", "ac_dc", "cable_in", "bus", "cable_out", "dc_ac", "hts_pow"]
    positions = {"gt": [5], "mot": [10, 15], "bus": 3, "mot_frac": [0.8, 0.2]}
    # TODO: add real positions

    N_motors = 2 * len(positions["mot"])
    N_turbines = 2 * len(positions["gt"])
    N_cables = N_motors + N_turbines

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
    components_with_losses = ["dc_ac", "bus", "ac_dc"]
    converter_sizing = size_all_converters(components_with_losses, comp=comp_params)
    print("Electrical components sizing complete.")

    max_powers, length = get_maximum_powers(powers)
    print(f"Wire length: {length}")

    cable_results = size_cables(max_powers, length=length, N_cables=N_cables, SF=2)