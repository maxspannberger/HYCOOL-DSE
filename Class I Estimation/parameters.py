"""
Parameters for the Class I estimation.
"""

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
    "Cruise_altitude": 6096.0,  # Cruise altitude [m]
    "MCR": 0.7,  # Cruise Mach number
    "TO_field_length": 1000,  # Take-off field length [m]
    "MTOW": 28000.0  # Maximum take-off weight [kg]
}
