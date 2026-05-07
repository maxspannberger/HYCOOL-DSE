import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# this file will contain all the classes that are used in this Class I estimation

class MatchingDiagram:
    def __init__(self, Vs0, Vapp, G, TO_field_length):
        
        self.Ne = 2 # Number of engines
        self.A = None # Aspect Ratio
        self.hcr = 20000 # Cruise altitude [ft]
        self.W_S = []

        # Requirements
        self.Vs0 = Vs0 # Stall speed
        self.Vapp = Vapp # Approach speed
        self.MCR = 0.7 # Cruise Mach number
        self.G = G # Climb Gradient
        self.TO_field_length = TO_field_length

    
    def calculate_matching(self, Vs0, Vapp, G, TO_field_length):
        """
        Calculate the matching diagram curves.
        """
        # 1. Stall Speed Requirement
        
        # Stall Speed 
        # Vs = np.sqrt((2(self.W_S))/())

        # 2. Take-off Distance Requirement


        # 3. Landing Distance Requirement


        # 4. Climb Requirements


        # 5. Maneuvering Requirements


        # 6. Cruise Requirements 

        # Put all of the curves together in a dictionary
        matching_curves = {
                "stall_speed_curve": None, # Placeholder for stall speed curve
                "takeoff_distance_curve": None,  # Placeholder for take-off distance curve
                "landing_distance_curve": None,  # Placeholder for landing distance curve
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
    


