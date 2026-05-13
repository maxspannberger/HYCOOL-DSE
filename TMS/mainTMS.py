import numpy as np

import pipe_python

# Values to input
P_cl = 5099.5
P_cr = 3792
P_res = 1281.8
P_OEI = 3100

# Other values
Charging_percent = 0.05
P_ch = (1 + Charging_percent) * P_cr
P_extra_1 = P_cl - P_ch
P_extra_2 = P_cl - P_cr
P_rem_1 = P_OEI - (P_ch / 2)
P_rem_2 = P_OEI - (P_cr / 2)

sp_work = 48

T_start = 20
T_use_fc = 433
T_use_gt = 318.9

###############################################################################
# Cooling and piping parameters
###############################################################################

# Hydrogen thermodynamic properties for coolant calculations.  The latent heat
# of vaporisation and approximate specific heat capacity of gaseous hydrogen
# are used to estimate how much thermal energy a kilogram of liquid
# hydrogen can absorb when warmed from cryogenic conditions up to the
# component inlet temperature.  Values are in kJ/kg and kJ/kg/K.
H_VAP = 446         # latent heat of vaporisation of LH2 [kJ/kg]
CP_GAS = 14.3       # specific heat of gaseous hydrogen [kJ/kg/K]
T_BOIL = 20.3       # approximate boiling temperature of LH2 [K]

###############################################################################
# Thermal management system calculations
###############################################################################

# Heat rejection fractions for each power source.  These values represent the
# fraction of the output power that must be rejected as waste heat.  While
# thermodynamic efficiency would suggest a larger waste fraction (e.g. a
# 60 % efficient fuel cell would reject 66.7 % of its output as heat), the
# user explicitly provided these fractions for sizing the cooling system.

HEAT_REJECTION_FRACTION = {
    "fc": 0.40,  # fuel cell: 40 % of input power is rejected as heat
    "bat": 0.10,  # battery: 10 % of input power is rejected as heat
    "gt": 0.089,  # gas turbine: 8.9 % of input power is rejected as heat
}

# Mapping of prime mover efficiencies.  These values represent the
# electrical/shaft efficiency of each device, not including additional
# electrical system losses.  For example, a gas turbine delivering power to
# an electric generator is roughly 39.5 % efficient, a fuel cell 60 %
# efficient and a battery 90 % efficient.  These are used to back‑calculate
# the upstream power requirement from the given output power.
EFFICIENCY_MAP = {
    "gt": 0.395,
    "fc": 0.60,
    "bat": 0.90,
}

# Electrical system loss fractions by source.  When electrical power is
# transmitted, a fraction of the power is lost as heat.  Gas turbine‑derived
# power incurs a 2.38 % penalty, whereas fuel cell or battery power incurs
# only a 1.72 % penalty.  When both gas turbines and other sources are
# present, the penalty applies per source (i.e. gas turbine output is
# penalised at 2.38 % and the remainder at 1.72 %).
PENALTY_MAP = {
    "gt": 0.0238,
    "fc": 0.0172,
    "bat": 0.0172,
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


def heat_absorption(power_kw: float, system: str, sp_work_mj_per_kg: float = sp_work) -> float:
    # Batteries do not consume hydrogen fuel, so they provide no cooling
    # capacity.  Their heat must be handled entirely by the available
    # coolant mass flow from other sources.
    if system == "bat":
        return 0.0
    # Convert specific work from MJ/kg to kJ/kg for consistency with the
    # enthalpy values (kJ/kg).  1 MJ = 1000 kJ.
    sp_kj_per_kg = sp_work_mj_per_kg * 1000.0
    # Determine efficiencies and electrical penalty for this system.  Use
    # defaults from the global maps; if the system is unknown, assume no
    # cooling capacity.
    eff = EFFICIENCY_MAP.get(system)
    penalty = PENALTY_MAP.get(system)
    if eff is None or penalty is None:
        return 0.0
    # Compute the chemical power required to deliver the desired output.
    # First account for electrical losses (1 - penalty), then for the prime
    # mover efficiency.  Example: for a gas turbine producing 1 kW of
    # useful power with 3.13 % electrical losses and 39.5 % efficiency, the
    # chemical power required is 1/(0.395*(1-0.0313)).
    power_input_kw = power_kw / (eff * (1.0 - penalty))
    # Mass flow rate of hydrogen (kg/s) needed to supply the required power.
    # 1 kW = 1 kJ/s, so dividing by sp_kj_per_kg gives kg/s.
    m_dot = power_input_kw / sp_kj_per_kg
    # Determine the enthalpy rise per kilogram based on the component type.
    if system == "gt":
        delta_h = H_VAP + CP_GAS * (T_use_gt - T_start)
    elif system == "fc":
        delta_h = H_VAP + CP_GAS * (T_use_fc - T_start)
    else:
        # Unknown system: no cooling capacity
        return 0.0
    # Heat absorbed = mass flow (kg/s) * enthalpy rise (kJ/kg) = kJ/s = kW
    return m_dot * delta_h


def compute_piping_losses(state_keys, states, design: str, flight_condition: str, sp_work_mj_per_kg: float = sp_work) -> float:
    # Convert specific work to kJ/kg
    sp_kj_per_kg = sp_work_mj_per_kg * 1000.0
    # Determine the power contributions from gas turbine and fuel cell
    # as well as their upstream power requirements.  Battery power does
    # not contribute to hydrogen mass flow.  Each output power is
    # back‑calculated to the chemical input power using the appropriate
    # efficiency and electrical penalty.
    power_input_gt_kw = 0.0
    power_input_fc_kw = 0.0
    for key in state_keys:
        sys = states[key]["system"]
        p_kw = states[key]["power_kw"]
        if sys == "gt":
            # Convert to chemical input power: account for
            # electrical losses and prime mover efficiency
            eff = EFFICIENCY_MAP["gt"]
            penalty = PENALTY_MAP["gt"]
            power_input_gt_kw += p_kw / (eff * (1.0 - penalty))
        elif sys == "fc":
            eff = EFFICIENCY_MAP["fc"]
            penalty = PENALTY_MAP["fc"]
            power_input_fc_kw += p_kw / (eff * (1.0 - penalty))
        # battery contributions do not affect hydrogen mass flow
    # Compute mass flow rates using input power (chemical) rather than output power
    m_dot_gt = power_input_gt_kw / sp_kj_per_kg
    m_dot_fc = power_input_fc_kw / sp_kj_per_kg
    # Determine the full and half flow lengths for this design
    if design == "A":
        full_length = 18.0
        half_length = 64.0
        extra_length = 0.0
    elif design == "B":
        full_length = 18.0
        half_length = 16.0
        extra_length = 0.0
    elif design == "C":
        full_length = 18.0
        half_length = 16.0
        extra_length = 0.0
    elif design == "D":
        # Full and half lengths for design D are fixed
        full_length = 18.0
        half_length = 16.0
        # Additional segment (14 m) carries fuel cell mass flow in climb and OEI
        extra_length = 14.0 if flight_condition in ("Climb", "OEI") else 0.0
    else:
        # Unknown design: no piping losses
        return 0.0
    total_heat_w = 0.0
    # Determine which mass flow should be applied to the primary piping network.
    # Designs A and C use gas turbines as the primary source; design B uses
    # fuel cells; design D uses gas turbines primarily and fuel cells only on
    # the additional short segment.
    if design == "B":
        m_dot_primary = m_dot_fc
    else:
        m_dot_primary = m_dot_gt
    # Full flow segment: hydrogen flows at full primary mass flow
    if m_dot_primary > 0.0 and full_length > 0.0:
        total_heat_w += pipe_python.run_pipe_analysis(m_dot_primary, full_length, 0)["total_heat_input_w"]
    # Half flow segment: carries half of the primary mass flow
    if m_dot_primary > 0.0 and half_length > 0.0:
        total_heat_w += pipe_python.run_pipe_analysis(m_dot_primary / 2.0, half_length, 0)["total_heat_input_w"]
    # Extra segment for design D when FC is present
    if extra_length > 0.0 and m_dot_fc > 0.0:
        total_heat_w += pipe_python.run_pipe_analysis(m_dot_fc, extra_length, 0)["total_heat_input_w"]
    return total_heat_w / 1000.0

def thermal_ratio_score(ratio):
    # Asymmetric margin bands (upper / lower tolerance around 1.0):
    #   Score 5: +0.25 / -0.50  -> [0.50, 1.25]
    #   Score 4: +0.50 / -1.00  -> [0.00, 1.50]
    #   Score 3: +0.75 / -1.00  -> [0.00, 1.75]
    #   Score 2: +1.00 / -1.00  -> [0.00, 2.00]
    #   Score 1: +1.25 / -1.00  -> [0.00, 2.25]
    if not np.isfinite(ratio):
        return 0
    if 0.50 <= ratio <= 1.25:
        return 5
    elif 0.00 <= ratio <= 1.50:
        return 4
    elif 0.00 <= ratio <= 1.75:
        return 3
    elif 0.00 <= ratio <= 2.00:
        return 2
    elif 0.00 <= ratio <= 2.25:
        return 1
    else:
        return 0

def design_phase_table() -> 'pd.DataFrame':
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
    design_conditions.append({
        "design": "A",
        "flight_condition": "OEI_bat",
        "states": ["p_oei_bat"],
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
    design_conditions.append({
        "design": "B",
        "flight_condition": "OEI_fc",
        "states": ["p_oei_fc"],
    })

    # Design C (two gas turbines)
    design_conditions.append({
        "design": "C",
        "flight_condition": "Cruise",
        "states": ["p_cr_gt"],
    })
    design_conditions.append({
        "design": "C",
        "flight_condition": "Climb",
        "states": ["p_cl_gt"],
    })
    design_conditions.append({
        "design": "C",
        "flight_condition": "Reserve",
        "states": ["p_res_gt"],
    })
    # For OEI in design C, one gas turbine remains, therefore the power
    # corresponds to the OEI condition for GT. It uses the state
    # 'p_oei_gt'.
    design_conditions.append({
        "design": "C",
        "flight_condition": "OEI",
        "states": ["p_oei_gt"],
    })

    # Design D (two gas turbines plus fuel cell)
    design_conditions.append({
        "design": "D",
        "flight_condition": "Cruise",
        "states": ["p_cr_gt"],
    })
    design_conditions.append({
        "design": "D",
        "flight_condition": "Climb",
        "states": ["p_cr_gt", "p_extra_fc"],
    })
    design_conditions.append({
        "design": "D",
        "flight_condition": "Reserve",
        "states": ["p_res_gt"],
    })
    # For OEI in design D, half of the charging cruise power comes from
    # the surviving gas turbine and the remainder from the fuel cell. This is
    # modelled by states 'p_cr_half_gt' (half of P_ch) and 'p_rem_fc'.
    design_conditions.append({
        "design": "D",
        "flight_condition": "OEI",
        "states": ["p_cr_half_gt", "p_rem_fc"],
    })
    design_conditions.append({
        "design": "D",
        "flight_condition": "OEI_gt",
        "states": ["p_oei_gt"],
    })
    
    # Build the results table
    rows = []
    for cond in design_conditions:
        power_total_kw = 0.0
        heat_total_kw = 0.0
        heat_abs_total_kw = 0.0
        power_gt_kw = 0.0
        power_other_kw = 0.0
        # Accumulate values over all states composing this flight condition
        for state_key in cond["states"]:
            state = states[state_key]
            system = state["system"]
            p_kw = state["power_kw"]
            # Sum the output power and heat rejection contributions
            power_total_kw += p_kw
            heat_total_kw += state["heat_kw"]
            # Tally power for electrical penalty calculation
            if system == "gt":
                power_gt_kw += p_kw
            else:
                power_other_kw += p_kw
            # Compute heat absorbed by hydrogen for this state
            heat_abs_total_kw += heat_absorption(p_kw, system)
        # Electrical system thermal contribution: apply 3.13 % to gas turbine power
        # and 1.83 % to all other power sources.  This heat adds to the component
        # heat that must be rejected.
        heat_elec_kw = power_gt_kw * PENALTY_MAP["gt"] + power_other_kw * PENALTY_MAP["fc"]
        # Add electrical heat into the total heat to reject from components
        heat_total_kw += heat_elec_kw
        # Compute piping losses (kW) based on the component mass flows, design and flight condition
        pipe_loss_kw = compute_piping_losses(cond["states"], states, cond["design"], cond["flight_condition"])
        # Sum heat rejection (including electrical contributions) and piping losses
        total_heat_kw = heat_total_kw + pipe_loss_kw
        # Ratio of heat to reject to heat absorbed; infinite if no absorption available
        ratio_rej_abs = (heat_total_kw / heat_abs_total_kw) if heat_abs_total_kw > 0 else float('inf')
        thermal_score = thermal_ratio_score(ratio_rej_abs)
        # Net heat is positive if there is still heat left to reject after hydrogen has absorbed all it can
        net_heat_kw = total_heat_kw - heat_abs_total_kw
        # Determine heat status: positive means there is still heat left to reject,
        # negative means the hydrogen cooling capacity exceeds the heat load.
        heat_status = "Positive" if net_heat_kw > 0 else "Negative"
        rows.append({
            "Design": cond["design"],
            "FlightCondition": cond["flight_condition"],
            # "States": "+".join(cond["states"]),
            "TotalPower_kW": power_total_kw,
            # "HeatElec_kW": heat_elec_kw,
            "HeatToReject_kW": heat_total_kw,
            # "PipingLoss_kW": pipe_loss_kw,
            "HeatAbsorbed_KW": heat_abs_total_kw,
            "RatioRejAbs": ratio_rej_abs,
            "ThermalScore": thermal_score,
            "NetHeat_kW": net_heat_kw,
            "HeatStatus": heat_status,
        })
    df = pd.DataFrame(rows)
    return df

def design_score_table(df=None):
    import pandas as pd

    if df is None:
        df = design_phase_table()

    weights = {
        "Cruise": 0.4,
        "Climb": 0.4,
        "OEI": 0.2,
    }

    rows = []

    for design in sorted(df["Design"].unique()):
        design_df = df[df["Design"] == design]

        cruise_score = design_df.loc[
            design_df["FlightCondition"] == "Cruise", "ThermalScore"
        ].iloc[0]

        climb_score = design_df.loc[
            design_df["FlightCondition"] == "Climb", "ThermalScore"
        ].iloc[0]

        oei_scores = design_df.loc[
            design_df["FlightCondition"].str.contains("OEI"), "ThermalScore"
        ]

        oei_score = oei_scores.min()

        final_score = (
            weights["Cruise"] * cruise_score
            + weights["Climb"] * climb_score
            + weights["OEI"] * oei_score
        )

        rows.append({
            "Design": design,
            "CruiseScore": cruise_score,
            "ClimbScore": climb_score,
            "OEIScore": oei_score,
            "FinalThermalScore": final_score,
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # When run as a script, compute and display the state table. This allows
    # quick testing from the command line (e.g. `python mainTMS.py`).
    import pandas as pd
    table = design_phase_table()
    scores = design_score_table(table)

    with pd.option_context("display.float_format", "{:,.2f}".format):
        print("\nDetailed thermal table:")
        print(table.to_string(index=False))

        print("\nDesign thermal scores:")
        print(scores.to_string(index=False))