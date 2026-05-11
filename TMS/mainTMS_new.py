import numpy as np
import matplotlib.pyplot as plt
from math import pi

P_cl = 5099.5
P_cr = 3792
Charging_percent = 0.05
P_ch = (1 + Charging_percent) * P_cr
P_res = 1281.8
P_extra_1 = P_cl - P_ch
P_extra_2 = P_cl - P_cr
P_OEI = 3100
P_rem_1 = P_OEI - (P_ch / 2)
P_rem_2 = P_OEI - (P_cr / 2)

sp_work = 48

T_start = 20
T_use_fc = 433
T_use_gt = 318.9

###############################################################################
# Thermal management system calculations
###############################################################################

# Heat rejection fractions for each power source. These values represent the
# fraction of the input energy that must be rejected as waste heat. For
# example, the fuel cell has an efficiency of 60 %, meaning that 40 % of its
# energy input is rejected as heat. The battery and gas turbine are similar.

HEAT_REJECTION_FRACTION = {
    "fc": 0.40,  # fuel cell: 40 % of input power is rejected as heat
    "bat": 0.10,  # battery: 10 % of input power is rejected as heat
    "gt": 0.089,  # gas turbine: 8.9 % of input power is rejected as heat
}


def heat_rejection(power_kw: float, system: str) -> float:
    if system not in HEAT_REJECTION_FRACTION:
        raise ValueError(f"Unknown system '{system}'. Must be one of {list(HEAT_REJECTION_FRACTION.keys())}.")
    fraction = HEAT_REJECTION_FRACTION[system]
    return power_kw * fraction


def compute_states() -> dict:
    states = {}
    # 1. p_ch from gt
    states["p_ch_gt"] = {
        "power_kw": P_ch,
        "system": "gt",
    }
    # 2. p_extra from bat
    states["p_extra_bat"] = {
        "power_kw": P_extra_1,
        "system": "bat",
    }
    # 3. p_res from gt
    states["p_res_gt"] = {
        "power_kw": P_res,
        "system": "gt",
    }
    # 4. p_oei from gt
    states["p_oei_gt"] = {
        "power_kw": P_OEI,
        "system": "gt",
    }
    # 5. p_oei from bat
    states["p_oei_bat"] = {
        "power_kw": P_OEI,
        "system": "bat",
    }
    # 6. p_ch from fc
    states["p_ch_fc"] = {
        "power_kw": P_ch,
        "system": "fc",
    }
    # 7. p_res from fc
    states["p_res_fc"] = {
        "power_kw": P_res,
        "system": "fc",
    }
    # 8. p_oei from fc
    states["p_oei_fc"] = {
        "power_kw": P_OEI,
        "system": "fc",
    }
    # 9. p_ch/2 for fc
    states["p_ch_half_fc"] = {
        "power_kw": P_ch / 2.0,
        "system": "fc",
    }
    # 10. p_rem for bat
    states["p_rem_bat"] = {
        "power_kw": P_rem_1,
        "system": "bat",
    }
    # 11. p_cr for gt
    states["p_cr_gt"] = {
        "power_kw": P_cr,
        "system": "gt",
    }
    # 12. p_cl for gt
    states["p_cl_gt"] = {
        "power_kw": P_cl,
        "system": "gt",
    }
    # 13. p_extra from fc
    states["p_extra_fc"] = {
        "power_kw": P_extra_2,
        "system": "fc",
    }
    # 14. p_cr/2 for gt
    states["p_cr_half_gt"] = {
        "power_kw": P_cr / 2.0,
        "system": "gt",
    }
    # 15. p_rem for fc
    states["p_rem_fc"] = {
        "power_kw": P_rem_2,
        "system": "fc",
    }
    # Compute heat rejection for each state
    for key, vals in states.items():
        power_kw = vals["power_kw"]
        system = vals["system"]
        vals["heat_kw"] = heat_rejection(power_kw, system)
    return states


def design_phase_table() -> 'pd.DataFrame':
    """
    Assemble a table (pandas DataFrame) describing each design and flight
    condition along with the total power demand and total heat rejection.

    This function uses the state definitions from compute_states() to build
    composite flight conditions for each design. If future piping losses
    need to be included, modify the `piping_losses_kw` variable or add
    further logic here.

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarising the flight conditions for each design.
    """
    import pandas as pd

    states = compute_states()

    # Placeholder for piping losses (kW). Modify this later when data is
    # available for each design or flight condition.
    piping_losses_kw = 0.0

    # Define the composition of each design and flight condition in terms of
    # the states computed above. Each entry is a list of state keys to sum.
    design_conditions = []

    # Design A
    design_conditions.append({
        "design": "A",
        "flight_condition": "Cruise",
        "states": ["p_ch_gt"],
    })
    design_conditions.append({
        "design": "A",
        "flight_condition": "Climb",
        "states": ["p_ch_gt", "p_extra_bat"],
    })
    design_conditions.append({
        "design": "A",
        "flight_condition": "Reserve",
        "states": ["p_res_gt"],
    })
    # For OEI in design A we assume the remaining gas turbine continues to
    # supply power. If desired, change 'p_oei_gt' to 'p_oei_bat' to model
    # battery‑only operation.
    design_conditions.append({
        "design": "A",
        "flight_condition": "OEI",
        "states": ["p_oei_gt"],
    })

    # Design B
    design_conditions.append({
        "design": "B",
        "flight_condition": "Cruise",
        "states": ["p_ch_fc"],
    })
    design_conditions.append({
        "design": "B",
        "flight_condition": "Climb",
        "states": ["p_ch_fc", "p_extra_bat"],
    })
    design_conditions.append({
        "design": "B",
        "flight_condition": "Reserve",
        "states": ["p_res_fc"],
    })
    # For OEI in design B, we split the power between the fuel cell (half of
    # the charging cruise power) and the battery supplying the remainder.
    design_conditions.append({
        "design": "B",
        "flight_condition": "OEI",
        "states": ["p_ch_half_fc", "p_rem_bat"],
    })

    # Design C (two gas turbines)
    design_conditions.append({
        "design": "C",
        "flight_condition": "Cruise",
        "states": ["p_ch_2gt"],
    })
    design_conditions.append({
        "design": "C",
        "flight_condition": "Climb",
        "states": ["p_cl_2gt"],
    })
    design_conditions.append({
        "design": "C",
        "flight_condition": "Reserve",
        "states": ["p_res_2gt"],
    })
    # For OEI in design C, one gas turbine remains, therefore the power
    # corresponds to the OEI condition for two GT. It uses the state
    # 'p_oei_2gt'.
    design_conditions.append({
        "design": "C",
        "flight_condition": "OEI",
        "states": ["p_oei_2gt"],
    })

    # Design D (two gas turbines plus fuel cell)
    design_conditions.append({
        "design": "D",
        "flight_condition": "Cruise",
        "states": ["p_ch_2gt"],
    })
    design_conditions.append({
        "design": "D",
        "flight_condition": "Climb",
        "states": ["p_ch_2gt", "p_extra_fc"],
    })
    design_conditions.append({
        "design": "D",
        "flight_condition": "Reserve",
        "states": ["p_res_2gt"],
    })
    # For OEI in design D, half of the charging cruise power comes from
    # the surviving gas turbine and the remainder from the fuel cell. This is
    # modelled by states 'p_ch_half_gt' (half of P_ch) and 'p_rem_fc'.
    design_conditions.append({
        "design": "D",
        "flight_condition": "OEI",
        "states": ["p_ch_half_gt", "p_rem_fc"],
    })

    # Build the results table
    rows = []
    for cond in design_conditions:
        power_total_kw = 0.0
        heat_total_kw = 0.0
        components = []
        for state_key in cond["states"]:
            state = states[state_key]
            power_total_kw += state["power_kw"]
            heat_total_kw += state["heat_kw"]
            components.append(state_key)
        # Add piping losses if any
        power_total_kw_with_losses = power_total_kw  # reserved for future
        heat_total_kw_with_losses = heat_total_kw + piping_losses_kw
        rows.append({
            "Design": cond["design"],
            "FlightCondition": cond["flight_condition"],
            "States": "+".join(cond["states"]),
            "TotalPower_kW": power_total_kw_with_losses,
            "HeatToReject_kW": heat_total_kw_with_losses,
        })
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # When run as a script, compute and display the state table. This allows
    # quick testing from the command line (e.g. `python mainTMS.py`).
    import pandas as pd
    table = design_phase_table()
    # Display the table in a readable format without scientific notation
    with pd.option_context('display.float_format', '{:,.2f}'.format):
        print(table.to_string(index=False))