"""
Parameters for the Class I estimation.
This file serves as the central database for all assumed constants and constraints.
"""
import numpy as np

# 1. Lift coefficients 
lift_coefficients = {
    "CL_max_cruise": 1.7,  # Lift coefficient during cruise
    "CL_max_TO": 1.9,      # Maximum lift coefficient during take-off
    "CL_max_L": 2.3        # Maximum lift coefficient during landing
}

# 2. Propulsion system parameters 
propulsion_parameters = {
    "Ne": 2,               # Number of engines [-]
    "eta_prop": 0.85,      # Propulsive efficiency [-]
    "eta_prop_ltr": 0.77,  # Propulsive efficiency during take-off [-]
    "kP": 0.321            # Propeller efficiency [-]
}

# 3. Aerodynamic parameters 
aerodynamic_parameters = {
    "CD0": 0.018,          # Zero-lift drag coefficient
    "e": 1.0,              # Oswald efficiency factor
    "A": 10.0,             # Aspect ratio
    "W_S_guess": 3500.0    # Initial wing loading guess [N/m^2] (Used for first iteration)
}

# 4. Flight parameters
flight_parameters = {
    "Cruise_altitude": 7620,     # Cruise altitude [m] (FL250)
    "MCR": 0.7,                  # Cruise Mach number
    "TO_field_length": 1000,     # Take-off field length [m]
    "MTOW": 28000.0,             # Maximum take-off weight [kg] (INITIAL GUESS)
    "cruise_range": 1000.0,      # Cruise range [km]
    "endurance_ltr": 0.75,       # Endurance [hours]
    "velocity_loiter": 65 * 1.3  # Loiter velocity [m/s]
}

# 5. Breguet Range Equation Parameters
breguet_parameters = {
    "L_D_Cruise": 12.0,  # Lift-to-drag ratio during cruise
    "cp_Cruise": 0.5,    # Specific fuel consumption [lbs/(hp*hr)]
    "L_D_ltr": 15.0,     # Lift-to-drag ratio during loiter
    "cp_ltr": 0.6,       # Specific fuel consumption during loiter [lbs/(hp*hr)]
}

# 6. Weight and Sizing Parameters (Required for Class I Iteration)
weight_parameters = {
    "payload_kg": 10000.0,        # Payload mass (e.g., 100 pax + baggage)
    "C_OE_guess": 0.58,          # Standard Empty Weight Fraction for Kerosene aircraft
    "contingency_margin": 0.0    # 5% FAR contingency fuel requirement
}

# 7. Mass Fractions (beta) based on Roskam empirical data
mass_fractions = { 
    "Mf_1": 0.99,   # Engine Start, Warm-up
    "Mf_2": 0.995,  # Taxi
    "Mf_3": 0.995,  # Take-off
    "Mf_4": 0.985,  # Climb
    # Cruise Fuel Fraction (Breguet)
    "Mf_5": np.exp(-(flight_parameters['cruise_range'] * 0.621371) / (375 * (propulsion_parameters['eta_prop']/breguet_parameters["cp_Cruise"]) * breguet_parameters["L_D_Cruise"])),  
    # Loiter Fuel Fraction (Breguet)
    "Mf_6": np.exp(-(flight_parameters['endurance_ltr'] * (flight_parameters['velocity_loiter']* 2.237)) / (375 * (propulsion_parameters['eta_prop_ltr']/breguet_parameters["cp_ltr"]) * breguet_parameters["L_D_ltr"])),  
    "Mf_7": 0.985,  # Descent
    "Mf_8": 0.995,  # Landing
}

def calculate_beta(mass_fractions, phase_string, loiter=False):
    """
    Calculates the accumulated weight fraction at a specific mission phase.
    Multiplies all mass fractions up to the requested phase.
    """
    mass_list = mass_fractions  
    phases = ["Mf_1", "Mf_2", "Mf_3", "Mf_4", "Mf_5", "Mf_6", "Mf_7", "Mf_8"]

    if not loiter:
        phases.remove("Mf_6")

    # Slice the list to only include phases up to the requested one
    phases = phases[:phases.index(phase_string) + 1]  
    
    beta_phase = 1.0
    for feiz in phases:
        beta_phase *= mass_list[feiz]

    return beta_phase

# Calculate the precise weight fractions remaining at each phase of the flight
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