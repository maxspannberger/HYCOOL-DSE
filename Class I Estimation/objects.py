import numpy as np
import scipy as sp
import math
import os
import matplotlib.pyplot as plt
from parameters import lift_coefficients

# this file will contain all the classes that are used in this Class I estimation

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
    

class MatchingDiagram:
    def __init__(self, Vs0=0, Vapp=57, LFL=1500, c=0, G=0.0024, TO_field_length=0):
        
        # Parameters
        self.Ne = 2 # Number of engines [-]
        self.A = 12 # Aspect Ratio [-]
        self.e = 0.8 # Oswald efficiency factor [-]
        self.hcr = 20000 # Cruise altitude [ft]
        self.beta = 0.7 # Mass Ratio [-]
        self.rho = 1.225 # Air density at sea level [kg/m^3]
        self.CD0 = 0.02 # Zero-lift drag coefficient [-]
        self.CLFL = 0.45 # Landing field length coefficient [-] (0.45 for CS25, 0.6 for CS23)
        self.gamma = Atmosphere.GAMMA
        self.R = Atmosphere.R
        self.h2 = 18 # Obstacle height [ft]
        self.kT = 0.85 # Take-off thrust parameter [-]
        self.g = Atmosphere.G
        self.S = Atmosphere.S
        self.MU_REF = Atmosphere.MU_REF
        self.T_REF = Atmosphere.T_REF
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

        # Arrays for PLotting
        self.W_S = np.linspace(100, 4000, 100)  # Example range for wing loading

    def calculate_matching(self, atm: Atmosphere):
        """
        Calculate the matching diagram curves.
        """
        # 1. Cruise Speed Requirement (TORENBEEK 156)
        T_W_cruise = (0.5 * self.gamma * np.square(self.MCR) * self.CD0) / (self.W_S / atm.pressure) + (self.W_S / atm.pressure) * (1 / (0.5 * self.gamma * np.square(self.MCR) * self.A * self.e))
        
        # 2. Take-off Distance Requirement (TORENBEEK 169)
        T_W_TO = self.mu_prime + 1 / ((1.159*(self.BFL - (self.Delta_S_TO/np.sqrt(atm.density/self.rho))))/(self.W_S/(atm.density*self.g*self.CL_2)+ self.h_to) - 2.7)
        
        # 3. Landing Distance Requirement (TORENBEEK 171)
        W_S_land = (1 / (self.W_land/self.W_TO)) * ((self.S_land/(self.f_land * self.h_land)) - 10) * ((self.h_land * atm.density * self.g * self.lift_coefficients['CL_max_L'])/(1.52/self.a_g + 1.69))

        # 4. Minimum Speed Requirement (TORENBEEK 166)
        #W_S_min = 0.5 * atm.density * self.lift_coefficients['CL_max_L'] * np.square(self.Vapp / 1.23)

        # 5. Climb Gradient Requirement (OEI) (TORENBEEK 161)
        #T_W_climb = self.g_climb + (0.5 * self.gamma * np.square(self.MCR) * self.CD0) / (self.W_S / atm.pressure) + (self.W_S / atm.pressure) * (1 / (0.5 * self.gamma * np.square(self.MCR) * np.pi * self.A * self.e))
        T_W_climb_min = self.g_climb + 2 * np.sqrt(self.CD0 / (np.pi * self.A * self.e))  
        # Generate an array the same size as W_S for the climb gradient requirement, where all values are equal to T_W_climb_min
        T_W_climb = np.full_like(self.W_S, T_W_climb_min)  

        # Put all of the curves together in a dictionary
        T_W_Curves = {
                "Cruise Speed Requirement": T_W_cruise,
                "Take-off Distance Requirement": T_W_TO,
                "Climb Gradient Requirement": T_W_climb
        }

        W_S_Curves = {
                "Landing Distance Requirement": W_S_land#,
           #     "Minimum Speed Requirement": W_S_min
        }

        return T_W_Curves, W_S_Curves
    
    def generate_diagram(self, T_W_Curves, W_S_Curves):
        """
        Generate the matching diagram based on the calculated performance parameters.
        """
        # Loop through the matching curves and plot them against the W_S values
        for curve_name, curve_data in T_W_Curves.items():
            # Plot the curve (this is a placeholder, actual plotting code will depend on the data structure of curve_data)
            plt.plot(self.W_S, curve_data, label=curve_name)
            pass
    
        for curve_name, curve_data in W_S_Curves.items():
            # Plot the curve (this is a placeholder, actual plotting code will depend on the data structure of curve_data)
            plt.vlines(curve_data, ymin=0, ymax=max(T_W_Curves["Cruise Speed Requirement"]), label=curve_name, linestyles='dashed')
            pass
        
        # Set the labels and title for the diagram
        plt.xlabel('Wing Loading (W/S)')
        plt.ylabel('Thrust Loading (T/W)')
        plt.title('HYCOOL Matching Diagram')
        plt.legend()

        figures_dir = os.path.join(os.path.dirname(__file__), 'Class_I_Figures')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, 'matching_diagram.png'))
        plt.show()

        return 

if __name__ == "__main__":
    TO_field_length = None

    atmosphere = Atmosphere(6096.0)  # Example altitude of 20,000 ft (6096 m)

    diagram = MatchingDiagram(TO_field_length)
    T_W_Curves, W_S_Curves = diagram.calculate_matching(atmosphere)
    print("T/W Curves:", T_W_Curves)
    print("W/S Curves:", W_S_Curves)
    diagram.generate_diagram(T_W_Curves, W_S_Curves)
