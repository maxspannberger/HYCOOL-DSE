from pathlib import Path
import sys
root = Path(__file__).resolve().parent
sys.path.append(str(root))

import insulation
import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
    

'''
This file contains classes that define all the components that can be present 
in the H2 infrastructure. Each component has a name and a position indicated 
with an index starting at 0, with 0 being the tank. 
'''

# =============================================================================
# Define the tank class
# =============================================================================
class Tank:
    def __init__(self, diameter: float,   wall: list,
                       name: str = 'Tank', position: int = 0):
        
        # initialise the tank specific parameters
        self.name = name
        self.position = position
        self.d = diameter
        self.wall = wall
  
# =============================================================================
# Define the pipe class
# =============================================================================
class Pipe:
    def __init__(self, position: int,    length: float, 
                       diameter: float,   wall: list, 
                       segments: int,     fluid: str = 'ParaHydrogen',  
                       name: str = 'Pipe'):
        
        # Initialise the pipe specific parameters
        self.name = name
        self.position = position
        self.fluid     = 'ParaHydrogen'
        self.length    = length
        self.segments  = segments
        self.wall      = wall
        self.cs        = 1 # STILL HAS TO BE IMPLEMENTED
        self.cr        = 1 # STILL HAS TO BE IMPLEMENTED
        self.cg        = 1 # STILL HAS TO BE IMPLEMENTED
        self.eps       = 1 # STILL HAS TO BE IMPLEMENTED
     
    # Function that loops over the pipe segments and tracks the state if H2
    def solve_H2_state(self, states, T_amb, m_dot, PLOT=False):
        p = states['p'][-1][-1] # Access the last pressure from the last component
        T = states['T'][-1][-1] # Access the last temperature from the last component
        h = CP.PropsSI('H', 'P', p, 'T', T, self.fluid)
        
        # Store results
        results = {'T':   np.zeros(self.segments), 
                   'p':   np.zeros(self.segments),
                   'rho': np.zeros(self.segments),
                   'h':   np.zeros(self.segments),
                   }
        
        # Loop over the pipe segments and adjust the state variables
        for i in range(self.segments):
            Q_dot = 1 # STILL HAS TO BE IMPLEMENTED function of T
            
            # Update state variables based on Q_dot and friction
            h  += (Q_dot / m_dot)
            p  += 300 # STILL HAS TO BE IMPLEMENTED BASED ON FRICTION
            T   = CP.PropsSI('T', 'P', p, 'H', h, self.fluid)
            rho = CP.PropsSI('D', 'P', p, 'H', h, self.fluid)
            
            # Store the updated state variables
            results['T'][i]   = T
            results['P'][i]   = p
            results['rho'][i] = rho
            results['h'][i]   = h
        
        # Plot the state variables allong the pipe.
        if PLOT:
                fig, axes = plt.subplots(2, 2, sharex=True)
                
                # Flatten axes for 0-3 indexing
                ax = axes.flatten()
                
                # Define plot data and styling
                plot_data = [
                    (results['T'], 'Temperature (K)', 'tab:red'),
                    (results['P'], 'Pressure (Pa)', 'tab:blue'),
                    (results['rho'], 'Density (kg/m³)', 'tab:green'),
                    (results['h'], 'Enthalpy (J/kg)', 'tab:purple')
                ]
                
                # Create a list of x positions. It starts after the first segment 
                # and ends at the end of the pipe
                x_values = np.linspace(0, self.length, self.segments + 1)[1:]
                
                # Iterate to fill the axis with the correct data
                for i, (data, label, color) in enumerate(plot_data):
                    ax[i].plot(x_values, data, color=color, linewidth=2)
                    ax[i].set_ylabel(label)
                    ax[i].grid(True, linestyle='--', alpha=0.7)
                
                # Tighten the layout, adjust the title size and plot
                fig.suptitle('H2 state accros pipe', fontsize=16)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
               
        # Return the results dictionary
        return results 
 
# =============================================================================
# Define the heat exchanger class             
# =============================================================================
class HEX:
    def __init__(self,
                 name:str, 
                 position: int,
                 catalyst: bool = False):
    
        self.name = name
        self.position = position