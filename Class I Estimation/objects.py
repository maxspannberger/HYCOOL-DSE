import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from parameters import lift_coefficients

# this file will contain all the classes that are used in this Class I estimation

class MatchingDiagram:
    def __init__(self, Vs0=0, Vapp=0, LFL=0, c=0, G=0, TO_field_length=0):
        
        # Parameters
        self.Ne = 2 # Number of engines [-]
        self.A = None # Aspect Ratio [-]
        self.e = None # Oswald efficiency factor [-]
        self.hcr = 20000 # Cruise altitude [ft]
        self.beta = 0.7 # Mass Ratio [-]
        self.rho = 1.225 # Air density at sea level [kg/m^3]
        self.CD0 = 0.02 # Zero-lift drag coefficient [-] 
        self.CLFL = 0.45 # Landing field length coefficient [-] (0.45 for CS25, 0.6 for CS23)
        self.gamma = 1.4 # Ratio of specific heats [-]
        self.R = 287 # Specific gas constant for air [J/(kg*K)]
        self.h2 = 18 # Obstacle height [ft]
        self.kT = 0.85 # Take-off thrust parameter [-]
        self.W_S = np.array([])
        self.g = 9.81
        self.lift_coefficients = lift_coefficients

        # Take-off Requirements
        self.h_to = 10.7  # [m]
        self.mu_prime = .010 * lift_coefficients['CL_max_TO'] + .02
        self.CL_2 = .694 * lift_coefficients['CL_max_TO']
        self.T_mean = None
        self.Delta_gamma_2 = None
        self.gamma_2 = None
        self.gamma_2_min = .024  # .024, .027, .030 for Ne = 2, 3, or 4 respectively
        self.BFL = 1800 # Balanced field length [m]
        self.Delta_S_TO = 200 # [m]

        # Requirements
        self.Vs0 = Vs0 # Stall speed 
        self.Vapp = Vapp # Approach speed
        self.LFL = LFL # Landing field length
        self.MCR = 0.7 # Cruise Mach number
        self.c = c # Climb rate
        self.G = G # Climb Gradient
        self.LTO = TO_field_length  # Take-off field length

        # Landing Requirements
        self.W_land = 25000  # [kg]
        self.S_land = 2200 # [m]
        self.W_TO = 32000 # [kg]
        self.f_land = 1.67  # CS 25
        self.h_land = 15.3  # CS 25 [m]
        self.a_g = 0.4  # CHECK TORENBEEK 170

        # Climb Requirements
        self.g_climb = 0.024  # Minimum climb gradient for OEI (2.4% for twin-engine aircraft)

    
    def calculate_matching(self, atm: Atmosphere):
        """
        Calculate the matching diagram curves.
        """
        # 1. Cruise Speed Requirement (TORENBEEK 156)
        T_W_cruise = (0.5 * self.gamma * np.square(self.MCR) * self.Cd0) / (self.W_S / atm.pressure) + (self.W_S / atm.pressure) * (1 / (0.5 * self.gamma * np.square(self.MCR) * self.A * self.e))
        
        # 2. Take-off Distance Requirement (TORENBEEK 169)
        T_W_TO = self.mu_prime + 1 / ((1.159*(self.BFL - (self.Delta_S_TO/np.sqrt(atm.dens/self.rho))))/(self.W_S/(atm.dens*self.g*self.CL_2)+ self.h_to) - 2.7)
        
        # 3. Landing Distance Requirement (TORENBEEK 171)
        W_S_land = (1 / (self.W_land/self.W_TO)) * ((self.S_land/(self.f_land * self.h_land)) - 10) * ((self.h_land * atm.dens * self.g * self.lift_coefficients['CL_max_L'])/(1.52/self.a_g + 1.69))

        # 4. Minimum Speed Requirement
        W_S_min = 0.5 * atm.dens * self.lift_coefficients['CL_max_L'] * np.square(self.Vapp / 1.23)

        # 5. Climb Gradient Requirement (OEI)
        T_W_climb = self.g_climb + (0.5 * self.g_climb * np.square(self.MCR) * self.CD0) / (self.W_S / atm.pressure) + (self.W_S / atm.pressure) * (1 / (0.5 * self.g_climb * np.square(self.MCR) * np.pi * self.A * self.e))
        

        #alpha_T_TO = None # Placeholder for thrust lapse ratio during take-off
        #T_W = (1 / alpha_T_TO) * (1.15 * np.sqrt((self.Ne/(self.Ne - 1))*(self.W_S/(self.LTO*self.kT*))) + (self.Ne/(self.Ne - 1)) * (4 * self.h2)/(self.LTO))

        # 3. Landing Distance Requirement
        #W_S_land = (1 / self.beta) * (self.rho / 2) * (self.LFL / self.CLFL) * self.lift_coefficients["CL_max_L"]

        # 4. Climb Rate Requirement (OEI)
        #alpha_T_climb = None # Placeholder for thrust lapse ratio during climb
        #T_W_climb = (self.Ne / (self.Ne - 1)) * (self.beta / alpha_T_climb) * (np.sqrt((np.square(self.c)/(self.beta * self.W_S * np.sqrt(self.CD0 * np.pi * self.A * self.e)))*(self.rho / 2)) + np.sqrt((4 * self.CD0) / (np.pi * self.A * self.e)))

        # 5. Climb gradient requirement
        #alpha_T_climb_gradient = None # Placeholder for thrust lapse ratio during climb gradient
        #T_W_climb_gradient = (self.Ne / (self.Ne - 1)) * (self.beta / alpha_T_climb_gradient) * (self.G *

        # 1. Minimum Speed Requirement
        W_S_min = (1 / self.beta) * (atm.rho / 2) * np.square(self.Vapp / 1.23) * self.lift_coefficients["CL_max_L"]
        
        # 2. Take-off Distance Requirement

        

        # 3. Landing Distance Requirement
        W_S_land = (1 / self.beta) * (self.rho / 2) * (self.LFL / self.CLFL) * self.lift_coefficients["CL_max_L"]

        # 4. Climb Rate Requirement (OEI)
        alpha_T_climb = None # Placeholder for thrust lapse ratio during climb
        T_W_climb = (self.Ne / (self.Ne - 1)) * (self.beta / alpha_T_climb) * (np.sqrt((np.square(self.c)/(self.beta * self.W_S * np.sqrt(self.CD0 * np.pi * self.A * self.e)))*(self.rho / 2)) + np.sqrt((4 * self.CD0) / (np.pi * self.A * self.e)))

        # 5. Climb gradient requirement
        alpha_T_climb_gradient = None # Placeholder for thrust lapse ratio during climb gradient
        T_W_climb_gradient = (self.Ne / (self.Ne - 1)) * (self.beta / alpha_T_climb_gradient) * (self.G * 2 * np.sqrt(self.CD0/(np.pi * self.A * self.e)))

        # 6. Maneuvering Requirements


        # 7. Cruise Speed Requirements 

        # Put all of the curves together in a dictionary
        matching_curves = {
                "minimum_speed_curve": W_S_min, # Placeholder for minimum speed curve
                "takeoff_distance_curve": None,  # Placeholder for take-off distance curve
                "landing_distance_curve": W_S_land,  # Placeholder for landing distance curve
                "climb_performance_curve": None,  # Placeholder for climb performance curve
                "maneuvering_performance_curve": None,  # Placeholder for maneuvering performance curve
                "cruise_performance_curve": None  # Placeholder for cruise performance curve
        }

        return matching_curves
    
    def generate_diagram(self, matching_curves):
        """
        Generate the matching diagram based on the calculated performance parameters.
        """
        # Loop through the matching curves and plot them against the W_S values
        for curve_name, curve_data in matching_curves.items():
            # Plot the curve (this is a placeholder, actual plotting code will depend on the data structure of curve_data)
            plt.plot(self.W_S, curve_data, label=curve_name)
            pass
        
        # Set the labels and title for the diagram
        plt.xlabel('Wing Loading (W/S)')
        plt.ylabel('Power Loading (W/P)')
        plt.title('Matching Diagram')
        plt.legend()
        plt.show()

        # Save the matching diagram to the figures folder
        plt.savefig('figures/matching_diagram.png')

        return 

class Atmosphere:
    """
    International Standard Atmosphere (ISA) model.
    Calculates atmospheric properties up to 20,000 meters.
    Example:
    cruise_std = Atmosphere(altitude)
    temp = cruise_std.temperature
    press = cruise_std.pressure
    dens = cruise_std.density
    sos = cruise_std.speed_of_sound
    """
    # Universal Constants
    G = 9.80665          # Gravity (m/s^2)
    R = 287.0528         # Specific gas constant for dry air (J/(kg·K))
    GAMMA = 1.4          # Specific heat ratio for air
    
    # Sutherland's Law Constants (for dynamic viscosity)
    S = 110.4            # Sutherland's temperature (K)
    MU_REF = 1.716e-5    # Reference dynamic viscosity (kg/(m·s))
    T_REF = 273.15       # Reference temperature (K)

    def __init__(self, altitude_m, isa_offset_c=0.0):
        """
        Initialize the atmospheric state.
        
        Args:
            altitude_m (float): Altitude above sea level in meters.
            isa_offset_c (float): Temperature deviation from standard day in Celsius/Kelvin.
                                  e.g., isa_offset_c=15 represents an ISA+15 Hot Day.
        """
        if altitude_m < 0:
            raise ValueError("Altitude must be zero or positive.")
            
        self.altitude = altitude_m
        self.isa_offset = isa_offset_c
        
        # Calculate standard properties upon initialization
        self._calculate_state()

    def _calculate_state(self):
        # --- Layer 0: Troposphere (0 to 11,000 m) ---
        if self.altitude <= 11000.0:
            Tb = 288.15
            Pb = 101325.0
            a = -0.0065
            
            self.temperature = Tb + a * self.altitude
            self.pressure = Pb * (self.temperature / Tb) ** (-self.G / (a * self.R))
            
        # --- Layer 1: Tropopause (11,000 m to 20,000 m) ---
        elif self.altitude <= 20000.0:
            Tb = 216.65
            Pb = 22632.1
            hb = 11000.0
            
            self.temperature = Tb
            self.pressure = Pb * math.exp(-self.G * (self.altitude - hb) / (self.R * Tb))
            
        else:
            raise NotImplementedError("Model only supports altitudes up to 20,000 meters.")
            
        # Apply the non-standard temperature offset (if any)
        self.temperature += self.isa_offset
        
        # --- Derived Properties ---
        
        # 1. Density (Ideal Gas Law)
        self.density = self.pressure / (self.R * self.temperature)
        
        # 2. Speed of Sound
        self.speed_of_sound = math.sqrt(self.GAMMA * self.R * self.temperature)
        
        # 3. Dynamic Viscosity (Sutherland's Law)
        self.dynamic_viscosity = (self.MU_REF * ((self.temperature / self.T_REF) ** 1.5) * ((self.T_REF + self.S) / (self.temperature + self.S)))
                                 
        # 4. Kinematic Viscosity (Dynamic viscosity / density)
        self.kinematic_viscosity = self.dynamic_viscosity / self.density

    def set_altitude(self, new_altitude_m):
        self.altitude = new_altitude_m
        self._calculate_state()

    def __repr__(self):
        return (f"<Atmosphere @ {self.altitude}m | "
                f"T: {self.temperature:.2f} K | "
                f"P: {self.pressure:.1f} Pa | "
                f"rho: {self.density:.4f} kg/m³>")

if __name__ == "__main__":
    TO_field_length = None
    diagram = MatchingDiagram(TO_field_length)
    diagram.generate_diagram(TO_field_length)    
