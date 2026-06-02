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
        
        # Initialise the tank specific parameters
        self.name = name
        self.position = position
        self.d = diameter
        self.wall = wall
        self.fluid = 'Hydrogen'
        
        # The state of the hydrogen in the tank
        self.p   = 6*101325
        self.T   = CP.PropsSI('T', 'P', self.p, 'Q', 0, self.fluid)
        self.rho = CP.PropsSI('D', 'P', self.p, 'Q', 0, self.fluid)
        self.h   = CP.PropsSI('H', 'P', self.p, 'Q', 0, self.fluid)
    
    def solve_H2_state(self, states, T_amb, m_dot, PLOT=False):
        
        # Store results
        results = {'T':   np.array([self.T]), 
                   'p':   np.array([self.p]),
                   'rho': np.array([self.rho]),
                   'h':   np.array([self.h])
                   }
        
        return results
            
# =============================================================================
# Define the pipe class
# =============================================================================
class Pipe:
    def __init__(self, position: int,    length: float,
                       diameter: float,   wall: list,
                       segments: int,     N: int,
                       N_bar: float,      P_mli: float,
                       eps_pipe: float    = 1.5e-5,
                       fluid: str         = 'Hydrogen',
                       name: str          = 'Pipe'):

        # Initialise the pipe specific parameters
        self.name = name
        self.position = position
        self.fluid     = fluid
        self.length    = length
        self.segments  = segments
        self.wall      = wall
        self.d         = diameter
        self.N         = N         # number of MLI layers
        self.N_bar     = N_bar     # layer density [layers/cm]
        self.P_mli     = P_mli     # residual gas pressure [Pa]
        self.eps_pipe  = eps_pipe  # pipe wall roughness [m]
        # Note: P_mli must be supplied in Pa; converted to Torr internally
        # because the Lockheed C_G constant was fitted with pressure in Torr
        self.cs        = 1.93 * 10**-6 # MLI conductivity coefficient [W/(m*K^(3.63))]
        self.cr        = 3.88 * 10**-10 # MLI radiation coefficient [W/(m^2*K^(4.67))]
        self.cg        = 5.5 * 10**4 # MLI gas conduction coefficient [W/(m^2*Pa*K^(0.52))]
        self.eps       = 0.03 # MLI emissivity, typical value for aluminized Mylar
     
    # Function that loops over the pipe segments and tracks the state if H2
    def solve_H2_state(self, states, T_amb, m_dot, PLOT=True):
        p = states['p'][-1][-1] # Access the last pressure from the last component
        T = states['T'][-1][-1] # Access the last temperature from the last component
        h = CP.PropsSI('H', 'P', p, 'Q', 0, self.fluid)
        
        # Store results
        results = {'T':   np.zeros(self.segments), 
                   'p':   np.zeros(self.segments),
                   'rho': np.zeros(self.segments),
                   'h':   np.zeros(self.segments),
                   }
        
        # Pre-compute segment geometry (constant along pipe)
        dz    = self.length / self.segments
        A_cs  = np.pi * self.d**2 / 4
        A_seg = np.pi * self.d * dz

        # Loop over the pipe segments and adjust the state variables
        for i in range(self.segments):
            # Fluid properties at segment inlet
            rho = CP.PropsSI('D', 'P', p, 'Q', 0, self.fluid)
            mu  = CP.PropsSI('V', 'P', p, 'Q', 0, self.fluid)

            # MLI heat leak (Lockheed three-term equation)
            T_h   = T_amb
            T_c   = T
            T_m   = (T_h + T_c) / 2
            Q_dot = A_seg * (
                (self.cs * T_m * self.N_bar**2.63 * (T_h - T_c)) / (self.N - 1)
              + (self.cr * self.eps * (T_h**4.67 - T_c**4.67)) / self.N
              + (self.cg * (self.P_mli / 133.322) * (T_h**0.52 - T_c**0.52)) / self.N
            )

            # Flow velocity and Reynolds number
            u  = m_dot / (rho * A_cs)
            Re = 4 * m_dot / (np.pi * self.d * mu)

            # Friction factor: Hagen-Poiseuille (laminar) or Haaland (turbulent)
            if Re < 2300:
                f = 64 / Re
            else:
                f = (1 / (-1.8 * np.log10((self.eps_pipe / self.d / 3.7)**1.11 + 6.9 / Re)))**2

            # Darcy-Weisbach pressure drop
            dp = f * (dz / self.d) * 0.5 * rho * u**2

            # Update state variables
            h  += Q_dot / m_dot
            p  -= dp
            T   = CP.PropsSI('T', 'P', p, 'H', h, self.fluid)
            rho = CP.PropsSI('D', 'P', p, 'H', h, self.fluid)
            
            # Store the updated state variables
            results['T'][i]   = T
            results['p'][i]   = p
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
                    (results['p'], 'Pressure (Pa)', 'tab:blue'),
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