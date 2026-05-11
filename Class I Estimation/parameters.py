"""
Parameters for the Class I estimation.
"""
import numpy as np

# 1. Lift coefficients 
lift_coefficients = {
    "CL_max_cruise": 1.4,  # Lift coefficient during cruise
    "CL_max_TO": 1.8,  # Maximum lift coefficient during take-off
    "CL_max_L": 2.4  # Maximum lift coefficient during landing
}

# 2. Propulsion system parameters 
propulsion_parameters = {
    "Ne": 2,  # Number of engines [-]
    "eta_prop": 0.85,  # Propulsive efficiency [-]
    "eta_prop_ltr": 0.77,  # Propulsive efficiency during take-off [-]r
    "kP": 0.321  # Propeller efficiency [-]
}

# 3. Aerodynamic parameters 
aerodynamic_parameters = {
    "CD0": 0.02,  # Zero-lift drag coefficient
    "e": 0.8,  # Oswald efficiency factor
    "A": 10.0  # Aspect ratio
}

# 4. Flight parameters
flight_parameters = {
    "Cruise_altitude": 7620,  # Cruise altitude [m] (FL250)
    "MCR": 0.7,  # Cruise Mach number
    "TO_field_length": 1000,  # Take-off field length [m]
    "MTOW": 28000.0,  # Maximum take-off weight [kg]
    "cruise_range": 1000.0,  # Cruise range [km]
    "endurance_ltr": 0.75,  # Endurance [hours]
    "velocity_loiter": 65 * 1.3  # Loiter velocity [m/s] #65 chosen for a landing or take-off 
}



# 6. Breguet
breguet_parameters = {
    "L_D_Cruise": 12.0,  # Lift-to-drag ratio during cruise
    "cp_Cruise": 0.5,  # Specific fuel consumption [kg/(N·s)]
    "L_D_ltr": 15.0, # Lift-to-drag ratio during take-off
    "cp_ltr": 0.6,  # Specific fuel consumption during take-off [kg/(N·s)]
}

# 5. Mass Fractions (beta)
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

def calculate_beta(mass_fractions, phase_string, loiter=False):
    """
    Calculate the mass fraction (beta) at a specified phase in the mission (phase_string).
    
    Parameters:
    mass_fractions (dict): A dictionary containing the mass fractions for each phase of the mission.
    phase_string (str): The phase of the mission for which to calculate the mass fraction.
    loiter (bool): Whether to consider the loiter phase. Default is False.
    
    Returns:
    float: The mass fraction (beta) for the specified phase.
    float: The overall mass fraction (beta) for the mission.
    """
    mass_lst = mass_fractions
    # Define the order of phases in the mission
    phases = ["Mf_1", "Mf_2", "Mf_3", "Mf_4", "Mf_5", "Mf_6", "Mf_7", "Mf_8"]

    # If loiter is not considered, exclude it from the phases
    if not loiter:
        phases.remove("Mf_6")

    phases = phases[:phases.index(phase_string) + 1]  # Include only phases up to the specified phase_string
    
    
    beta_phase = 1.0
    # Calculate the mass fraction for the specified phase
    for feiz in phases:
        beta_phase *= mass_lst[feiz]

    return beta_phase

beta_taxi = calculate_beta(mass_fractions, "Mf_2")
beta_cruise = calculate_beta(mass_fractions, "Mf_5")
beta_landing = calculate_beta(mass_fractions, "Mf_8", loiter=True)

print(beta_taxi)
print(beta_cruise)
print(beta_landing)


"""
# Multiply all mass fractions to get the fuel mass fraction for the mission
Mff = 1.0
for key in mass_fractions:
    Mff *= mass_fractions[key]

print(Mff)
"""