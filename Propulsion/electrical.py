import numpy as np
import sys
from pathlib import Path
import json

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
import os

from General.component_parameters import component_params as comp_params
from WeightEstimations.Aircraft_Config import default_q400_hycool
# from WeightEstimations.mainClassII import run_class_ii


def get_cable_region_powers(component, positions, previous, b=1.0):
    cable_loss_per_m = 0.0 # kW/m

    comp_powers = {}
    maximum = {}
    maximum[positions["gt"][0]] = {}

    if "in" in component:
        positions_gt = positions["gt"] + [-x for x in positions["gt"]]
        for pos in positions_gt:
            comp_powers[pos] = {}
            comp_powers[pos]["length"] = abs(pos - positions["bus"]) * b/2

            for condition in previous[list(previous.keys())[0]]:
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
            comp_powers[pos]["length"] = abs(pos - positions["bus"]) * b/2

            for condition in previous[list(previous.keys())[0]]:                
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
                elif condition == "OEI_gt":
                    maximum[positions["gt"][0]][condition] = max(previous[abs(pos)][condition] / frac * 2, maximum[positions["gt"][0]][condition])
                else:
                    maximum[positions["gt"][0]][condition] = max(previous[abs(pos)][condition] / frac, maximum[positions["gt"][0]][condition])

    else:
        raise ValueError("There are only 2 conditions for cabling: before bus (1) and after bus (2).")

    return comp_powers, maximum


def get_powers_per_component(P_TO, P_climb, P_cruise, P_APP, P_OEI, P_AC_systems, positions, component_order, comp=comp_params, b=1.0):
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
                "TO": P_TO / 2 * frac,
                "climb": P_climb / 2 * frac,
                "cruise": P_cruise / 2 * frac,
                "APP": P_APP / 2 * frac,
                "OEI_mot": P_OEI / (1 + min(positions["mot_frac"])) * frac,
                "OEI_gt": P_OEI / 2 * frac,
                "OEI_bus": P_OEI  * frac / 2
            }

    previous = comp_powers["out"].copy()

    # -------------------------------------------------------------
    # Walk upstream through architecture
    # ------------------------------------------------------------

    for component in reversed(component_order):
        comp_powers[component] = {}

        if "cable" in component:
            comp_powers_cable, previous = get_cable_region_powers(component, positions, previous, b=b)
            comp_powers[component] = comp_powers_cable

        else:
            for pos in previous:
                comp_powers[component][pos] = {}
                for condition in previous[pos]:
                    comp_powers[component][pos][condition] = previous[pos][condition] / comp[component].efficiency
                if component == "bus":
                    if condition == "OEI_bus":
                        comp_powers[component][pos][condition] += P_AC_systems / comp[component].efficiency
                    else:
                        comp_powers[component][pos][condition] += 0.5 * P_AC_systems / comp[component].efficiency
            previous = comp_powers[component].copy()

    filename = os.path.join(root, "Propulsion", "power_chain_results.json")
    with open(filename, "w") as f:
        json.dump(comp_powers, f, indent=4)

    return comp_powers


def get_Rth(P, eff, T_J, T_H2):
    Q = (1-eff) * P
    R_th = (T_J - T_H2) / Q
    return R_th


def get_deltaT(P, eff, R_th):
    Q = (1-eff) * P
    deltaT  = Q * R_th
    return deltaT


def get_P_idle(T_J, R_th, T_H2):
    P = (T_J - T_H2) / R_th
    return P


def size_converter(component, powers, comp=comp_params, show=False):    
    T_H2 = 20   # K
    T_J_design = 250    # K
    T_J_min = 77    # K

    converter_proportions = (0.3, 0.5, 0.2)
    prop_scaling = (0.3 * 0.5 * 0.2) ** (1.0/3.0)

    eff = comp[component].efficiency
    P_OEI = max(powers["OEI_mot"], powers["OEI_gt"])
    R_th = get_Rth(powers["TO"], eff, T_J_design, T_H2)
    T_J_cruise = T_H2 + get_deltaT(powers["cruise"], eff, R_th)
    T_J_OEI = T_H2 + get_deltaT(P_OEI, eff, R_th)
    P_heat = get_P_idle(T_J_min, R_th, T_H2)

    P_max = max(P_OEI, powers["TO"])
    P_cool = P_max * (1 - eff)

    mass = P_max / comp[component].power_density
    volume = P_max / comp[component].volumetric_density

    base_side_length = volume ** (1.0/3.0)
    converter_sizes = tuple([base_side_length * l/prop_scaling for l in converter_proportions])

    if show:
        print(f"Number of chips: {R_th}")
        print(f"Cruise temperature: {T_J_cruise}")
        print(f"OEI temperature: {T_J_OEI}")
        print(f"Idle extra heat: {P_heat}")
        print(f"Max cooling power: {P_cool}")
        print(f"Mass: {mass}")

    results = {
        "R_th": R_th,
        "T_J_cruise": T_J_cruise,
        "T_J_OEI": T_J_OEI,
        "P_heat": P_heat,
        "P_cool": P_cool,
        "mass": mass,
        "volume": volume,
        "sizes": converter_sizes
    }

    return results


def size_all_components(component_order, powers, HTS_dimensions, comp=comp_params, show=False):
    component_sizing = {}
    P_heat_total = 0.0
    P_cool_total = 0.0
    m_total = 0.0

    for component in component_order:
        if "gt" not in component and "cable" not in component:
            component_sizing[component] = {}

            for pos in powers[component]:
                component_sizing[component][pos] = {}
                if "hts" in component:                  
                    P_max = max(powers[component][pos].values())
                    component_sizing[component][pos]["P_cool"] = (1.0 - comp[component].efficiency) * P_max
                    component_sizing[component][pos]["mass"] = P_max / comp[component].power_density
                    if "gen" in component:
                        component_sizing[component][pos]["sizes"] = (HTS_dimensions["hts_gen"]["L"], HTS_dimensions["hts_gen"]["D"])
                    else:
                        if np.isclose(pos, 1.0):
                            component_sizing[component][pos]["sizes"] = (HTS_dimensions["hts_pow_2"]["L"], HTS_dimensions["hts_pow_2"]["D"])
                        else:
                            component_sizing[component][pos]["sizes"] = (HTS_dimensions["hts_pow_1"]["L"], HTS_dimensions["hts_pow_1"]["D"])

                    P_cool_total += component_sizing[component][pos]["P_cool"] * 2
                    m_total += component_sizing[component][pos]["mass"]

                else:
                    if show:
                        print(f"\n{component} ({pos}):")
                    component_sizing[component][pos] = size_converter(component, powers[component][pos], comp=comp, show=show)
                    P_heat_total += component_sizing[component][pos]["P_heat"] * 2
                    P_cool_total += component_sizing[component][pos]["P_cool"] * 2
                    m_total += component_sizing[component][pos]["mass"] * 2
        
    component_sizing["total"] = {
        "P_heat": P_heat_total,
        "P_cool": P_cool_total,
        "mass": m_total
    }

    filename = os.path.join(root, "Propulsion", "component_sizing_results.json")
    with open(filename, "w") as f:
        json.dump(component_sizing, f, indent=4)

    dimensions_only = {}
    for component in component_sizing:
        if component != "total":
            dimensions_only[component] = {}
            for loc in component_sizing[component]:
                dimensions_only[component][loc] = component_sizing[component][loc]["sizes"]

    filename = os.path.join(root, "Propulsion", "only_sizing_results.json")
    with open(filename, "w") as f:
        json.dump(dimensions_only, f, indent=4)

    cooling_requirements_only = {}
    for condition in list(list(powers.values())[0].values())[0].keys():
        cooling_requirements_only[condition] = {}
        total = 0.0
        for component in component_order:
            if "gt" not in component and "cable" not in component:
                cooling_requirements_only[condition][component] = {}
                for pos in powers[component]:
                    cooling_requirements_only[condition][component][pos] = (1.0 - comp[component].efficiency) * powers[component][pos][condition]

                    if condition == "OEI_gt" and component in ["hts_gen", "ac_dc"]:
                        total += cooling_requirements_only[condition][component][pos]
                    elif condition == "OEI_mot" and component in ["dc_ac", "hts_pow"] and not np.isclose(pos, 1.0):
                        total += cooling_requirements_only[condition][component][pos]
                    elif condition == "OEI_bus" and component in ["bus"]:
                        total += cooling_requirements_only[condition][component][pos]
                    else:
                        total += 2 * cooling_requirements_only[condition][component][pos]

        cooling_requirements_only[condition]["total"] = total

    filename = os.path.join(root, "Propulsion", "only_cooling_results.json")
    with open(filename, "w") as f:
        json.dump(cooling_requirements_only, f, indent=4)


    return component_sizing, cooling_requirements_only


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
                        if "in" in component and pos < 0.0:
                            length += powers[component][pos][condition] # account for auxiliary cable
            else:
                if component not in max_powers:
                    max_powers[component] = {}
                if pos not in max_powers[component]:
                    max_powers[component][pos] = 0.0
                for condition in powers[component][pos]:
                    if powers[component][pos][condition] > max_powers[component][pos]:
                        max_powers[component][pos] = powers[component][pos][condition]

    filename = os.path.join(root, "Propulsion", "max_power_results.json")
    with open(filename, "w") as f:
        json.dump(max_powers, f, indent=4)

    length *= 2 # only counted cables connected to one bus so far

    return max_powers, length


def size_cables(max_powers, length=200, N_cables=6, SF=1, show=False):
    V = 3000 # V
    rho_c = 6380 # kg/m^3
    rho_i = 900 # kg/m^3
    J_0 = 2.5e9 # A/m^2
    dJdB = -0.07 # exponent
    U_i = 5e7 # V/m
    mu_r = 2.6 # -
    mu_0 = np.pi*4e-7 # T*m/A

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
    
    if show:
        print(f"\ncable:")
        print(f"Cable length [m]: {length}")
        print(f"Optimal wire radius [mm]: {1000*r_c}")
        print(f"Optimal insulator thickness [mm]: {1000*t_i}")

    t_i = round(0.0001 * np.ceil(10000 * t_i), 4)
    r_c = round(0.00005 * np.ceil(20000 * r_c), 4)
    d = round(2 * (r_c + t_i), 4)
    A_c = np.pi * r_c**2
    A_i = np.pi * t_i * (2*r_c + t_i)
    mass_density = A_c * rho_c + A_i * rho_i
    mass = mass_density * length

    if show:
        print(f"Conservative wire radius [mm]: {1000*r_c}")
        print(f"Conservative insulator thickness [mm]: {1000*t_i}")
        print(f"Wire diameter [mm]: {1000*d}")
        print(f"Cable mass density [kg/m]: {mass_density:.6f}")
        print(f"Cable mass [kg]: {mass:.3f}")

    results = {
        "length": length,
        "r_core": r_c,
        "t_insulation": t_i,
        "d_cable": d,
        "m_density": mass_density,
        "m": mass
    }

    filename = os.path.join(root, "Propulsion", "cable_results.json")
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    return results


def size_APU(P_transient, P_base, component, comp=comp_params, show=False):
    if hasattr(comp[component], "energy_density"):
        P_total = (P_transient + P_base) / np.sqrt(comp[component].efficiency)
    else:
        P_total = P_transient + P_base
    mass = P_total / comp[component].power_density

    if hasattr(comp[component], "energy_density"):
        energy = mass * comp[component].energy_density
        time = energy * np.sqrt(comp[component].efficiency) / P_base * 60
        P_cool = (1 - np.sqrt(comp[component].efficiency)) * P_total
    else:
        time = np.inf
        P_cool = 0.0

    if show:
        print(f"\nAPU mass [kg]: {mass}")
        print(f"APU run time [min]: {time}")
        print(f"APU peak cooling power required [kW]: {P_cool}")

    APU_results = {
        "mass": mass,
        "time": time,
        "P_cool": P_cool
    }

    filename = os.path.join(root, "Propulsion", "APU_results.json")
    with open(filename, "w") as f:
        json.dump(APU_results, f, indent=4)

    return APU_results





def perform_complete_electrical_sizing(P_TO, P_climb, P_cruise, P_APP, P_OEI, b, show=False):
    P_AC_systems = 315.0  # kW

    # define electrical system architecture
    component_order = ["gt_hex", "hts_gen", "ac_dc", "cable_in", "bus", "cable_out", "dc_ac", "hts_pow"]
    positions = {"gt": [0.5], "mot": [0.5, 1.0], "bus": 0.5, "mot_frac": [0.8, 0.2]}
    apu = "bt"

    # HTS dimensions:
    HTS_dimensions = {
        "hts_gen": {
            "L": 0.5333,
            "D": 0.30
        },
        "hts_pow_1": {
            "L": 0.4244,
            "D": 0.30
        },
        "hts_pow_2": {
            "L": 0.24,
            "D": 0.20
        },
    }

    N_motors = 2 * len(positions["mot"])
    N_turbines = 4 * len(positions["gt"])
    N_cables = N_motors + N_turbines

    # perform sizing of electrical system
    powers = get_powers_per_component(P_TO, P_climb, P_cruise, P_APP, P_OEI, P_AC_systems, positions, component_order, comp=comp_params, b=b)
    converter_sizing, cooling_requirements_only = size_all_components(component_order, powers, HTS_dimensions, comp=comp_params, show=show)
    max_powers, length = get_maximum_powers(powers)
    cable_results = size_cables(max_powers, length=length, N_cables=N_cables/2, SF=2, show=show)
    APU_results = size_APU(converter_sizing["total"]["P_heat"], P_AC_systems, component=apu, comp=comp_params, show=True)

    if show:
        print("\nElectrical components sizing complete.")

    APU_mass = APU_results["mass"]

    total_mass = converter_sizing["total"]["mass"] + cable_results["m"]

    return total_mass, cooling_requirements_only


if __name__ == "__main__":
    # get class II power results
    print("Performing Class II estimations...")
    cfg = default_q400_hycool()
    class_II_results = run_class_ii(config=3, comp=comp_params, verbose=False, cfg=cfg)
    print("Class II estimations finished.")

    P_TO = class_II_results.P_takeoff_KW
    P_climb = class_II_results.mission.P_climb_shaft/1000.0
    P_cruise = class_II_results.mission.P_cruise_shaft/1000.0
    P_APP = class_II_results.P_approach_KW
    P_OEI = class_II_results.weight.P_TO_OEI_KW
    b = class_II_results.Wing_span

    mass, cooling_requirements = perform_complete_electrical_sizing(P_TO, P_climb, P_cruise, P_APP, P_OEI, b, show=True)