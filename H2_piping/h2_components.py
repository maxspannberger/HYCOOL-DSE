from pathlib import Path
import sys
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from General.component_parameters import component_params as comp

import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
    

'''
===============================================================================
This file contains classes that define all the components that can be present 
in the H2 infrastructure. Each component has a name and a position indicated 
with an index starting at 0, with 0 being the tank by default. 
===============================================================================
'''

# In order to track the gas fraction, this function outputs whether the coolprop
# value should be used or a manually picked value should be used 
# (mainly important for supercritical phases)
# =============================================================================
def calc_frac(p, h, fluid='Hydrogen'):
    # Determine the phase
    phase = CP.PhaseSI('P', p, 'H', h, fluid)
    
    if phase == 'twophase':
        return CP.PropsSI('Q', 'P', p, 'H', h, fluid)
    elif phase == 'liquid':
        return 0.0
    elif phase == 'gas' or phase == 'supercritical_gas':
        return 1.0
    else:
        raise ValueError(f"Unknown phase '{phase}' encountered at p={p:.2f}, h={h:.2f}. "
                         "Check if input states are within physical limits.")
# =============================================================================

# Extract the most recent state value from the states dictionary
# =============================================================================
def get_input_states(states):
    T   = states['T'][-1][-1]
    p   = states['p'][-1][-1]
    h   = states['h'][-1][-1]
    rho = states['rho'][-1][-1]
    
    return T, p, h, rho
# =============================================================================

# Define an iterative solver for the state change over a pipe segment
# =============================================================================
def update_states(vars, p1, h1, m_dot, A_cs, fluid, q=0, dp_fric=0):
    rho1 = CP.PropsSI('D', 'P', p1, 'H', h1, fluid)
    u1   = m_dot / (rho1 * A_cs)
    
    p2, h2 = vars
    
    try:
        rho2 = CP.PropsSI('D', 'P', p2, 'H', h2, fluid)
        u2   = m_dot / (rho2 * A_cs)
    except: 
        return [1e9, 1e9] # Reset guess values if the solver diverges
    
    res_momentum   = (p2 + rho2 * u2**2) - (p1 + rho1 * u1**2) + dp_fric
    res_energy     = (h2 + 0.5 * u2**2) - (h1 + 0.5 * u1**2 + q)
    
    return [res_momentum, res_energy]
    
    
# =====================================================================


# =============================================================================
# Define the tank class
# =============================================================================
class Tank:
    def __init__(self, diameter: float,   
                       wall:     list,
                       position: int = 0):
        
        # Initialise the tank specific parameters
        self.name = 'Tank'
        self.position = position
        self.d = diameter
        self.wall = wall
        self.fluid = 'Hydrogen'
        
        # The state of the hydrogen in the tank based on the assumption that
        # the fluid is fully in a liquid state
        self.p   = 5*101325.
        self.frac= 0.
        self.T   = CP.PropsSI('T', 'P', self.p, 'Q', self.frac, self.fluid)
        self.rho = CP.PropsSI('D', 'P', self.p, 'Q', self.frac, self.fluid)
        self.h   = CP.PropsSI('H', 'P', self.p, 'Q', self.frac, self.fluid)
      
    # Set the tank values as the initial values of the piping system
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False):
        A_cs = np.pi * system[1].d**2 / 4
        u = m_dot / (self.rho * A_cs)
        h = self.h - 0.5 * u**2
        
        p    = self.p -0.5 * self.rho * u**2
        rho  = CP.PropsSI('D', 'P', p, 'H', h, self.fluid)
        T    = CP.PropsSI('T', 'P', p, 'H', h, self.fluid)
        frac = calc_frac(p, h)
        if frac > 0.01:
            raise ValueError(f"The hydrogen turns partially gasseous ({frac}) as it leaves "
                             "the tank. Incompressability assumption doesn't hold.")
        
        # Store results in dictionary and return
        results = {'T':   np.array([self.T, T]), 
                   'p':   np.array([self.p, p]),
                   'rho': np.array([self.rho, rho]),
                   'h':   np.array([self.h, h]),
                   'frac':np.array([self.frac, frac])
                   }
        
        return results
            
# =============================================================================
# Define the pipe class
# =============================================================================
class Pipe:
    def __init__(self, position: int,     
                       length:   float,
                       diameter: float,   
                       wall:     list,
                       segments: int,     
                       N:        int,
                       N_bar:    float,      
                       P_mli:    float,
                       curv:     float,
                       eps_pipe: float   = 1.5e-5,
                       fluid:    str     = 'Hydrogen'
                       ):

        # Initialise the pipe specific parameters
        self.name      = 'Pipe'
        self.position  = position
        self.fluid     = fluid
        self.length    = length
        self.segments  = segments
        self.wall      = wall
        self.d         = diameter
        self.N         = N         # number of MLI layers
        self.N_bar     = N_bar     # layer density [layers/cm]
        self.P_mli     = P_mli     # residual gas pressure [Pa]
        self.eps_pipe  = eps_pipe  # pipe wall roughness [m]
        self.curv      = curv      # curvature of the bends, R/d
        # Note: P_mli must be supplied in Pa; converted to Torr internally
        # because the Lockheed C_G constant was fitted with pressure in Torr
        self.cs        = 1.93 * 10**-6 # MLI conductivity coefficient [W/(m*K^(3.63))]
        self.cr        = 3.88 * 10**-10 # MLI radiation coefficient [W/(m^2*K^(4.67))]
        self.cg        = 5.5 * 10**4 # MLI gas conduction coefficient [W/(m^2*Torr*K^(0.52))], H2 (= N2 value 1.46e4 * sqrt(M_N2/M_H2))
        self.eps       = 0.03 # MLI emissivity, typical value for aluminized Mylar
     
    # Function that loops over the pipe segments and tracks the state if H2
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False):
        
        # Get the input state variables
        T1, p1, h1, rho1 = get_input_states(states)
        
        # Initialize the results dictionary
        results = {'T':   np.zeros(self.segments), 
                   'p':   np.zeros(self.segments),
                   'rho': np.zeros(self.segments),
                   'h':   np.zeros(self.segments),
                   'frac':np.zeros(self.segments)}
        
        # Pre-compute segment geometry assuming a constant pipe diam. and wall
        dz    = self.length / self.segments
        A_cs  = np.pi * self.d**2 / 4
        A_seg = np.pi * self.d * dz

        # Loop over the pipe segments and adjust the state variables
        for seg in range(self.segments):
            # Fluid properties at segment inlet
            mu1  = CP.PropsSI('V', 'P', p1, 'H', h1, self.fluid)

            # Flow velocity and Reynolds number
            u1  = m_dot / (rho1 * A_cs)
            Re1 = 4 * m_dot / (np.pi * self.d * mu1)
            
            # Calculate different T, such as they are presented in the
            # Lockhead equation
            T_h = T_amb
            T_c = T1
            T_m = (T_h + T_c) / 2
            
            # MLI heat leak equation Lockhead
            Q_dot = A_seg * (
                (self.cs * T_m * self.N_bar**2.63 * (T_h - T_c)) / (self.N - 1)
              + (self.cr * self.eps * (T_h**4.67 - T_c**4.67)) / self.N
              + (self.cg * (self.P_mli / 133.322) * (T_h**0.52 - T_c**0.52)) / self.N
            )
            
            # Determine the Friction factor:
            # either Hagen-Poiseuille (laminar) or Haaland (turbulent)
            if Re1 < 2300:
                f = 64 / Re1
            else:
                f = (1 / (-1.8 * np.log10((self.eps_pipe / self.d / 3.7)**1.11 + 6.9 / Re1)))**2
            
            # Update the enthalpy based on the heat leak
            q       = Q_dot / m_dot
            dp_fric = f * (dz / self.d) * 0.5 * rho1 * u1**2
            
            p2, h2 = fsolve(update_states,
                          x0=[p1, h1],
                          args=(p1, h1, m_dot, A_cs, self.fluid, q, dp_fric))
            
            # Update the state variables
            T2     = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
            rho2   = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
            frac2  = calc_frac(p2, h2, fluid='Hydrogen')            
            
            # Store the updated state variables
            results['T'][seg]   = T2
            results['p'][seg]   = p2
            results['rho'][seg] = rho2
            results['h'][seg]   = h2
            results['frac'][seg]= frac2
        
        if PLOT:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
            ax = axes.flatten()
            
            # Define plot data and styling
            plot_data = [
                (results['T'], 'Temperature (K)', 'tab:red'),
                (results['p'], 'Pressure (Pa)', 'tab:blue'),
                (results['rho'], 'Density (kg/m³)', 'tab:green'),
                (results['h'], 'Enthalpy (J/kg)', 'tab:purple')
            ]
            
            x_values = np.linspace(0, self.length, self.segments)
            
            # Iterate to fill the axis
            for i, (data, label, color) in enumerate(plot_data):
                # Plot the main data
                ax[i].plot(x_values, data, color=color, linewidth=2)
                ax[i].set_ylabel(label)
                ax[i].grid(True, linestyle='--', alpha=0.5)
            
            fig.suptitle('H2 State Across Pipe', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            
        return results
    
# =============================================================================
# Define the pipe bend class
# =============================================================================
class Corner:
    def __init__(self, position: int,
                       curv:     float,
                       diameter: float,
                       fluid:    str    = 'Hydrogen',
                       name:     str    = 'Bend',
                       N_bend:   int    = 1
                       ):
        
        self.position = position
        self.N_bend = N_bend
        self.curv = curv
        self.d = diameter
        self.fluid = fluid
        self.name = name
    
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False):
        # Extract states from end of previous component
        T1, p1, h1, rho1 = get_input_states(states)
        
        # Calculate viscocity and reqnolds number   
        mu1  = CP.PropsSI('V', 'P', p1, 'H', h1, self.fluid)
        Re1  = 4 * m_dot / (np.pi * self.d * mu1)
        
        # Calculat the pipe cross-sectional area and the speed
        A_cs = np.pi * self.d**2 / 4
        u1 = m_dot / (rho1 * A_cs)
        
        # Calculate the bend geometry factor
        alpha = 0.95 + 4.42 * (self.curv)**(-1.96)
        
        # Calculate K_bend
        K_bend = 0.388 * alpha * (self.curv)**0.84 * Re1**(-0.17)
        
        # Calculate pressure drop (using K_bend)
        dp_fric = K_bend * self.N_bend * 0.5 * rho1 * u1**2
        q       = 0
        
        # Update pressure and enthalpy
        p2, h2 = fsolve(update_states,
                      x0=[p1, h1],
                      args=(p1, h1, m_dot, A_cs, self.fluid, q, dp_fric))
        
        T2    = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        rho2  = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        frac2 = calc_frac(p2, h2, fluid='Hydrogen')
        
        # Store results
        results = {'T':    np.array([T2]), 
                   'p':    np.array([p2]),
                   'rho':  np.array([rho2]),
                   'h':    np.array([h2]),
                   'frac': np.array([frac2])
                   }
        
        return results

# =============================================================================
# Define the HTS generator/motor class
# =============================================================================
class HTS:
    def __init__(self,
                 power:      float,
                 name:       str     = 'HTS', 
                 efficiency: float   = comp['hts_gen'].efficiency
                 ):

        def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False):
            # Extract states from end of previous component
            T, p, h, rho = get_input_states(states)
            
            Q_dot = self.power * (1 - self.efficiency)
            dh    = Q_dot / m_dot
            h    += dh        
                       
            frac = calc_frac(p, h)
            
            # Store results in dictionary and return
            results = {'T':   np.array([T]), 
                       'p':   np.array([p]),
                       'rho': np.array([rho]),
                       'h':   np.array([h]),
                       'frac':np.array([frac])
                       }
            
            return results
        