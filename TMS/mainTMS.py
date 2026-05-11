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

def hydrogen_mass_flow(power, sp_work=48000):
    m_dot_h2 = power / sp_work  # [kg/s]
    return m_dot_h2

def hydrogen_heat_absorption(power, system_type, sp_work=48000):
    # Temperatures [K]
    T_start = 20
    T_boil = 20.3
    T_use_fc = 433
    T_use_gt = 318.9

    # Hydrogen properties
    h_vap = 446       # [kJ/kg]
    cp_gas = 14.3     # [kJ/kg/K]

    # Hydrogen mass flow
    m_dot_h2 = hydrogen_mass_flow(power, sp_work)

    if system_type == "GT":
        T_use = T_use_gt
    elif system_type == "FC":
        T_use = T_use_fc
    else:
        raise ValueError("system_type must be either 'GT' or 'FC'")

    # Heat absorbed per kg of hydrogen
    delta_h = h_vap + cp_gas * (T_use - T_boil)  # [kJ/kg]
    # Total heat absorption
    Q_abs = m_dot_h2 * delta_h  # [kW]

    return Q_abs, delta_h, m_dot_h2