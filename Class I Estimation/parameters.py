"""
Parameters for the Class I estimation.
"""
import numpy as np

# 1. Lift coefficients 
lift_coefficients = {
    "CL_max_cruise": 1.4,  
    "CL_max_TO": 1.8,  
    "CL_max_L": 2.4  
}

# 2. Propulsion system parameters 
propulsion_parameters = {
    "Ne": 2,  
    "eta_prop": 0.85,  
    "eta_prop_ltr": 0.77,  
    "kP": 0.321  
}

# 3. Aerodynamic parameters 
aerodynamic_parameters = {
    "CD0": 0.02,  
    "e": 0.8,  
    "A": 10.0,
    "W_S_guess": 1990.0  # Initial wing loading guess [kg/m^2]
}

# 4. Flight parameters
flight_parameters = {
    "Cruise_altitude": 7620,  
    "MCR": 0.7,  
    "TO_field_length": 1000,  
    "MTOW": 28000.0,  
    "cruise_range": 1000.0,  
    "endurance_ltr": 0.75,  
    "velocity_loiter": 65 * 1.3  
}

# 5. Breguet
breguet_parameters = {
    "L_D_Cruise": 12.0,  
    "cp_Cruise": 0.5,  
    "L_D_ltr": 15.0, 
    "cp_ltr": 0.6,  
}

# 6. Mass Fractions (beta)
mass_fractions = { # ROSKAM
    "Mf_1": 0.99,  # Engine Start, Warm-up
    "Mf_2": 0.995,  # Taxi
    "Mf_3": 0.995,  # Take-off
    "Mf_4": 0.985,  # Climb
    "Mf_5": np.exp(-(flight_parameters['cruise_range'] * 0.621371) / (375 * (propulsion_parameters['eta_prop']/breguet_parameters["cp_Cruise"]) * breguet_parameters["L_D_Cruise"])),  # Cruise
    "Mf_6": np.exp(-(flight_parameters['endurance_ltr'] * (flight_parameters['velocity_loiter']* 2.237)) / (375 * (propulsion_parameters['eta_prop_ltr']/breguet_parameters["cp_ltr"]) * breguet_parameters["L_D_ltr"])),  # Loiter
    "Mf_7": 0.985,  # Descent
    "Mf_8": 0.995,  # Landing
}

# 7. Weight and Sizing Parameters (BASELINE KEROSENE SIZING)
weight_parameters = {
    "payload_kg": 10000,  # 100 pax + baggage
    "C_OE_guess": 0.58,    # Standard Empty Weight Fraction for Kerosene aircraft
    "contingency_margin": 0.05 # 5% FAR contingency fuel requirement
}

def calculate_beta(mass_fractions, phase_string, loiter=False):
    phases = ["Mf_1", "Mf_2", "Mf_3", "Mf_4", "Mf_5", "Mf_6", "Mf_7", "Mf_8"]
    if not loiter:
        phases.remove("Mf_6")
    phases = phases[:phases.index(phase_string) + 1]  
    beta_phase = 1.0
    for feiz in mass_fractions:
        beta_phase *= mass_fractions[feiz]
    return beta_phase