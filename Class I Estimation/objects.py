import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from parameters import lift_coefficients

# this file will contain all the classes that are used in this Class I estimation

class MatchingDiagram:
    def __init__(self, Vs0, Vapp, LFL, c, G, TO_field_length):
        
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
        self.lift_coefficients = lift_coefficients

        # Requirements
        self.Vs0 = Vs0 # Stall speed 
        self.Vapp = Vapp # Approach speed
        self.LFL = LFL # Landing field length
        self.MCR = 0.7 # Cruise Mach number
        self.c = c # Climb rate
        self.G = G # Climb Gradient
        self.LTO = TO_field_length  # Take-off field length

    
    def calculate_matching(self):
        """
        Calculate the matching diagram curves.
        """
        # 1. Minimum Speed Requirement
        W_S_min = (1 / self.beta) * (self.rho / 2) * np.square(self.Vapp / 1.23) * self.lift_coefficients["CL_max_L"]
        
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


if __name__ == "__main__":
    TO_field_length = None
    diagram = MatchingDiagram(TO_field_length)
    diagram.generate_diagram(TO_field_length)
    


