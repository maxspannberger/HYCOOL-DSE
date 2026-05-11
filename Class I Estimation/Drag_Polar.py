import numpy as np
import matplotlib.pyplot as plt

class DragPolarEstimator:
    """
    Class to calculate and plot aerodynamic drag characteristics of an aircraft.
    """
    def __init__(self, W_kg, S, b, Cd0, e, rho=1.225):
        """
        Initialize the aircraft and atmospheric parameters.
        """
        self.W = W_kg
        self.S = S
        self.b = b
        self.Cd0 = Cd0
        self.e = e
        self.rho = rho
        
        # Derived parameter (Aspect Ratio)
        self.A = (self.b ** 2) / self.S
        
        # Initialize lists to store calculation results
        self.V_knots = None
        self.Cl_lst = []
        self.Cd_lst = []
        self.D_lst = []

    def _lift_coeff(self, V_ms):
        """Calculates Lift Coefficient (Cl) for a given velocity in m/s."""
        return (self.W * 9.80665) / (0.5 * self.rho * (V_ms ** 2) * self.S)

    def _induced_drag(self, Cl):
        """Calculates Induced Drag Coefficient (Cdi) for a given Cl."""
        return (Cl ** 2) / (np.pi * self.e * self.A)

    def calculate_performance(self, V_lst_knots):
        """
        Iterates over a list of velocities (in knots) to calculate Cl, Cd, and Total Drag.
        """
        self.V_knots = V_lst_knots
        
        # Reset lists in case the method is run multiple times
        self.Cl_lst = []
        self.Cd_lst = []
        self.D_lst = []
        
        for V_kt in self.V_knots:
            V_ms = V_kt * 0.5144444  # Convert knots to m/s
            
            # Aerodynamic state calculations
            Cl = self._lift_coeff(V_ms)
            Cdi = self._induced_drag(Cl)
            Cd = self.Cd0 + Cdi
            D = Cd * 0.5 * self.rho * (V_ms ** 2) * self.S
            
            # Store data for plotting
            self.Cl_lst.append(Cl)
            self.Cd_lst.append(Cd)
            self.D_lst.append(D)

    def plot_results(self):
        """Generates the Drag vs Velocity and Drag Polar plots."""
        if self.V_knots is None:
            raise ValueError("Data not found. Run 'calculate_performance()' before plotting.")
            
        # 1. Total Drag vs Velocity Plot
        plt.figure(figsize=(8, 5))
        plt.plot(self.V_knots, self.D_lst, color='b', linewidth=2)
        plt.title("Total Drag vs Velocity")
        plt.xlabel("Velocity (knots)")
        plt.ylabel("Total Drag (N)")
        plt.grid(True)
        
        # 2. Drag Polar Plot
        plt.figure(figsize=(8, 5))
        plt.plot(self.Cd_lst, self.Cl_lst, color='r', linewidth=2)
        plt.title("Drag Polar")
        plt.xlabel("Drag Coefficient ($C_D$)")
        plt.ylabel("Lift Coefficient ($C_L$)")
        plt.grid(True)
        
        plt.show()

# Protection block: only runs if Drag_Polar.py is executed directly.
if __name__ == "__main__":
    V_lst = np.linspace(20, 180, 100)
    
    # Instantiate the aircraft with dummy values for testing
    my_aircraft = DragPolarEstimator(W_kg=31000, S=64, b=28.4, Cd0=0.02, e=0.75)
    
    # Calculate and Plot
    my_aircraft.calculate_performance(V_lst)
    my_aircraft.plot_results()