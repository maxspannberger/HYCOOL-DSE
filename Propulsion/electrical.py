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
R_th_26 = 14.3  # K/kW
R_th = R_th_26/26

cable_loss_per_m = 0.0 # kW/m
P_AC_systems = 0.0  # kW


def get_cable_region_powers(P_out_branch, positions, condition):
    """
    Calculate power carried by every cable region in the aircraft.

    Parameters
    ----------
    P_total : float
        Total propulsion power required [kW]

    positions : dict
        Example:
        {
            "gt": [5],
            "mot": [10, 15]
        }

        Values are half-wing positions [m].

    condition : str
        "max"
        "cruise"
        "OEI_mot"
        "OEI_gt"

    Returns
    -------
    regions : list(dict)
        Example:
        [
            {
                "span": (-15,-10),
                "length": 5,
                "power": 1000,
                "loss": 5
            },
            ...
        ]

    total_loss : float
        Sum of all cable losses [kW]
    """

    # --------------------------------------------------
    # Build full-aircraft component locations
    # --------------------------------------------------
    full_gt_positions = (
        [-x for x in reversed(positions["gt"])]
        + positions["gt"]
    )
    full_mot_positions = (
        [-x for x in reversed(positions["mot"])]
        + positions["mot"]
    )

    N_gt = len(full_gt_positions)
    N_mot = len(full_mot_positions)

    if condition == "OEI_mot":
        active_gt = N_gt
        active_mot = N_mot - 1

    elif condition == "OEI_gt":
        active_gt = N_gt - 1
        active_mot = N_mot

    else:
        active_gt = N_gt
        active_mot = N_mot

    nodes = full_mot_positions + [0.0]
    node_losses = {x: 0.0 for x in nodes}
    old_loss = np.inf
    total_loss = 0.0
    i = 0

    while np.abs(total_loss - old_loss) > 1e-6 and i < 100:
        old_loss = total_loss
        i += 1

        P_in_branch = (P_out_branch * active_mot + P_AC_systems + total_loss) / active_gt 

        # --------------------------------------------------
        # Build injection list
        # + generation
        # - load
        # --------------------------------------------------
        injections = []

        # --------------------------
        # Turbines
        # --------------------------
        gt_locations = full_gt_positions.copy()
        if condition == "OEI_gt":
            # remove first turbine
            # (later you can replace with worst-case selection)
            gt_locations.pop(0)
        for x in gt_locations:
            injections.append((x, P_in_branch))

        # --------------------------
        # Motors
        # --------------------------
        mot_locations = full_mot_positions.copy()

        if condition == "OEI_mot":
            # remove first motor
            # (later replace by worst-case motor)
            mot_locations.pop(0)

        for x in mot_locations:
            P_load = P_out_branch + node_losses.get(x, 0.0)
            injections.append((x, -P_load))

        # --------------------------
        # AC systems at fuselage
        # --------------------------
        P_out_system = P_AC_systems + node_losses.get(0.0, 0.0)
        injections.append((0.0, -P_out_system))

        # --------------------------------------------------
        # Sort by spanwise position
        # --------------------------------------------------
        injections.sort(key=lambda item: item[0])

        # --------------------------------------------------
        # Power flow through each region
        # Flow in region i:
        # absolute cumulative injection
        # to the left of the region.
        # --------------------------------------------------

        power_flow = 0.0
        regions = []
        total_loss = 0.0
        node_losses = {x: 0.0 for x in node_losses}

        for i in range(len(injections) - 1):
            x_left = injections[i][0]
            x_right = injections[i + 1][0]
            power_flow += injections[i][1]

            length = abs(x_right - x_left)
            loss = (
                abs(power_flow)
                * cable_loss_per_m
                * length
            )
            total_loss += loss

            regions.append(
                {
                    "span": (x_left, x_right),
                    "length": length,
                    "power": power_flow,
                    "loss": loss,
                }
            )

            if power_flow > 1e-6:
                node_distances = [node - x_right for node in nodes]
                x_out = x_right + min([node for node in node_distances if node >= -1e-6])
                node_losses[x_out] += loss
            elif power_flow < -1e-6:
                node_distances = [node - x_left for node in nodes]
                x_out = x_left + max([node for node in node_distances if node <= 1e-6])
                node_losses[x_out] += loss

    return regions, total_loss


def get_powers_per_component(P_max, P_cruise, P_OEI, positions, component_order, comp):
    """
    Returns powers required at every component
    and cable segment powers.
    """

    N_motors = len(positions["mot"]) * 2
    N_turbines = len(positions["gt"]) * 2

    operating_conditions = {
        "max": P_max,
        "cruise": P_cruise,
        "OEI_mot": P_OEI,
        "OEI_gt": P_OEI,
    }

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

        # ---------------------------------------------------------
        # Cable section
        # ---------------------------------------------------------
        if component == "cable":
            comp_powers["cable"] = {}
            cable_losses = {}

            for condition, power_required in previous.items():
                regions, total_loss = get_cable_region_powers(
                    P_out_branch=power_required,
                    positions=positions,
                    condition=condition
                )
                comp_powers["cable"][condition] = {
                    "regions": regions,
                    "total_loss": total_loss,
                }
                cable_losses[condition] = total_loss

            # Add cable losses to required upstream power
            previous = {
                cond: previous[cond] + cable_losses[cond]
                for cond in previous
            }

            # Adjust for turbine/motor count change
            # across electrical distribution system

            previous["max"] *= N_motors / N_turbines
            previous["cruise"] *= N_motors / N_turbines
            previous["OEI_mot"] *= (
                (N_motors - 1) / N_turbines
            )
            previous["OEI_gt"] *= (
                N_motors / (N_turbines - 1)
            )

        # ---------------------------------------------------------
        # Efficiency component
        # ---------------------------------------------------------
        else:
            comp_powers[component] = {}
            for condition in previous:
                comp_powers[component][condition] = (
                    previous[condition]
                    / comp[component].efficiency
                )
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

def size_converter(comp, P_max, P_cruise, P_OEI):
    eff = comp.efficiency

    N = get_N(P_max, eff, T_J_design)
    T_J_cruise = T_H2 + get_deltaT(P_cruise, eff, N)
    T_J_OEI = T_H2 + get_deltaT(P_OEI, eff, N)
    T_J_idle = T_H2 + get_deltaT(0.0, eff, N)

    return N, T_J_cruise, T_J_OEI, T_J_idle


if __name__ == "__main__":
    component_order = ["gt_hex", "hts_gen", "ac_dc", "cable", "dc_ac", "hts_pow"]
    positions = {"gt": [5], "mot": [10, 15]}

    cfg = default_q400_hycool()
    class_II_results = run_class_ii(config=3, comp=comp_params, verbose=False, cfg=cfg)

    P_max = class_II_results.P_max_KW
    P_cruise = class_II_results.mission.P_cruise_shaft/1000.0
    P_OEI = class_II_results.weight.P_TO_OEI_KW


    powers = get_powers_per_component(P_max, P_cruise, P_OEI, positions, component_order, comp=comp_params)
    print(powers)

    filename = "power_chain_results.json"
    with open(filename, "w") as f:
        json.dump(powers, f, indent=4)



    inverter = comp_params["dc_ac"]
    rectifier = comp_params["ac_dc"]