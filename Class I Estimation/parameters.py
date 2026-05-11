"""
Parameters for the Class I estimation.
"""
import numpy as np

# 1. Lift coefficients 
lift_coefficients = {
    "CL_max_cruise": 1.7,  
    "CL_max_TO": 1.9,  
    "CL_max_L": 2.2  
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
    "e": 1.0,  
    "A": 10.0,
    "W_S_guess": 350.0 
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
mass_fractions = { 
    "Mf_1": 0.99,  
    "Mf_2": 0.995, 
    "Mf_3": 0.995, 
    "Mf_4": 0.985, 
    "Mf_5": np.exp(-(flight_parameters['cruise_range'] * 0.621371) / (375 * (propulsion_parameters['eta_prop']/breguet_parameters["cp_Cruise"]) * breguet_parameters["L_D_Cruise"])),  
    "Mf_6": np.exp(-(flight_parameters['endurance_ltr'] * (flight_parameters['velocity_loiter']* 2.237)) / (375 * (propulsion_parameters['eta_prop_ltr']/breguet_parameters["cp_ltr"]) * breguet_parameters["L_D_ltr"])),  
    "Mf_7": 0.985,  
    "Mf_8": 0.995,  
}

# 7. Weight and Sizing Parameters 
weight_parameters = {
    "payload_kg": 10000.0,
    "C_OE_guess": 0.58,  
    "contingency_margin": 0.05 
}

def calculate_beta(mass_fractions, phase_string, loiter=False):
    mass_list = mass_fractions  
    phases = ["Mf_1", "Mf_2", "Mf_3", "Mf_4", "Mf_5", "Mf_6", "Mf_7", "Mf_8"]

    if not loiter:
        phases.remove("Mf_6")

    phases = phases[:phases.index(phase_string) + 1]  
    
    beta_phase = 1.0
    for feiz in phases:
        beta_phase *= mass_list[feiz]

    return beta_phase

beta_dict = {
    'beta_engine_start': calculate_beta(mass_fractions, "Mf_1"),
    'beta_taxi': calculate_beta(mass_fractions, "Mf_2"),
    'beta_takeoff': calculate_beta(mass_fractions, "Mf_3"),
    'beta_climb': calculate_beta(mass_fractions, "Mf_4"),
    'beta_cruise': calculate_beta(mass_fractions, "Mf_5"),
    'beta_loiter': calculate_beta(mass_fractions, "Mf_6", loiter=True),
    'beta_descent': calculate_beta(mass_fractions, "Mf_7"), 
    'beta_descent_loiter': calculate_beta(mass_fractions, "Mf_7", loiter=True), 
    'beta_landing': calculate_beta(mass_fractions, "Mf_8"),
    'beta_landing_loiter': calculate_beta(mass_fractions, "Mf_8", loiter=True)
}