from pathlib import Path
import sys
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from component_parameters import component_params as comp

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

# Extract the most recent state value from the states dictionary
def get_input_states(states):
    T   = states['T'][-1][-1]
    p   = states['p'][-1][-1]
    h   = states['h'][-1][-1]
    rho = states['rho'][-1][-1]
    
    return T, p, h, rho

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
    def solve_H2_state(self, states, T_amb, m_dot, PLOT=False):
        
        # Store results in dictionary and return
        results = {'T':   np.array([self.T]), 
                   'p':   np.array([self.p]),
                   'rho': np.array([self.rho]),
                   'h':   np.array([self.h]),
                   'frac':np.array([self.frac])
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
    def solve_H2_state(self, states, T_amb, m_dot, PLOT=False):
        T, p, h, rho = get_input_states(states)
        
        # Define an iterative solver for the state change over a pipe segment
        # =====================================================================
        def solve_segment(p1, h1, rho1, m_dot, A_cs, q, fluid):
            # Calculate inlet density and velocity
            u1 = m_dot / (rho1 * A_cs)
            
            # Define the system of equations as a function
            def equations(vars):
                p2, h2 = vars
                
                # Get properties from the Equation of State at the proposed outlet state
                try:
                    rho2 = CP.PropsSI('D', 'P', p2, 'H', h2, fluid)
                    u2 = m_dot / (rho2 * A_cs)
                except:
                    return [1e6, 1e6] # Penalty for non-physical states
                    
                # Calculate residual from conservation of Energyq
                res_energy = (h2 + 0.5 * u2**2) - (h1 + 0.5 * u1**2 + q)
                
                # Calculate residual from conservation of momentum
                res_momentum = (p2 + rho2 * u2**2) - (p1 + rho1 * u1**2)
                
                return [res_energy, res_momentum]
        
            # Take the input parameters as a initial guess
            initial_guess = [p1, h1 + q]
            
            # Solve using scipy
            p2, h2 = fsolve(equations, initial_guess)
            return p2, h2
        # =====================================================================
        
        # Initialize the results dictionary
        results = {'T':   np.zeros(self.segments), 
                   'p':   np.zeros(self.segments),
                   'rho': np.zeros(self.segments),
                   'h':   np.zeros(self.segments),
                   'frac':np.zeros(self.segments)}
        
        # Pre-compute segment geometry (constant along pipe)
        dz    = self.length / self.segments
        A_cs  = np.pi * self.d**2 / 4
        A_seg = np.pi * self.d * dz

        # Loop over the pipe segments and adjust the state variables
        for seg in range(self.segments):
            # Fluid properties at segment inlet
            mu  = CP.PropsSI('V', 'P', p, 'H', h, self.fluid)

            # Flow velocity and Reynolds number
            u  = m_dot / (rho * A_cs)
            Re = 4 * m_dot / (np.pi * self.d * mu)
            
            # Calculate different T, such as they are presented in the
            # Lockhead equation
            T_h = T_amb
            T_c = T
            T_m = (T_h + T_c) / 2
            
            # MLI heat leak equation Lockhead
            Q_dot = A_seg * (
                (self.cs * T_m * self.N_bar**2.63 * (T_h - T_c)) / (self.N - 1)
              + (self.cr * self.eps * (T_h**4.67 - T_c**4.67)) / self.N
              + (self.cg * (self.P_mli / 133.322) * (T_h**0.52 - T_c**0.52)) / self.N
            )
            
            # Determine the Friction factor:
            # either Hagen-Poiseuille (laminar) or Haaland (turbulent)
            if Re < 2300:
                f = 64 / Re
            else:
                f = (1 / (-1.8 * np.log10((self.eps_pipe / self.d / 3.7)**1.11 + 6.9 / Re)))**2
            
            f = 0
            # Update the enthalpy based on the heat leak
            q = Q_dot / m_dot
            dp_friction = f * (dz / self.d) * 0.5 * rho * u**2
            
            p, h = solve_segment(p, h, rho, m_dot, A_cs, q, self.fluid)
            
            # Calculate the pressure change due to thermal expansion and friction
            dp_friction = f * (dz / self.d) * 0.5 * rho * u**2
            
            # Update the state variables
            p    -=  dp_friction
            T     = CP.PropsSI('T', 'P', p, 'H', h, self.fluid)
            rho   = CP.PropsSI('D', 'P', p, 'H', h, self.fluid)
            frac  = calc_frac(p, h, fluid='Hydrogen')            
            
            # Store the updated state variables
            results['T'][seg]   = T
            results['p'][seg]   = p
            results['rho'][seg] = rho
            results['h'][seg]   = h
            results['frac'][seg]= frac
        
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
    
    def solve_H2_state(self, states, T_amb, m_dot, PLOT=False):
        # Extract states from end of previous component
        T, p, h, rho = get_input_states(states)
        s = CP.PropsSI('S', 'P', p, 'H', h, self.fluid)
        
        # Calculate viscocity and reqnolds number   
        mu  = CP.PropsSI('V', 'P', p, 'H', h, self.fluid)
        print(mu)
        Re  = 4 * m_dot / (np.pi * self.d * mu)
        
        # Calculat the pipe cross-sectional area and the speed
        A_cs = np.pi * self.d**2 / 4
        u = m_dot / (rho * A_cs)
        
        # Calculate the bend geometry factor
        alpha = 0.95 + 4.42 * (self.curv)**(-1.96)
        
        # Calculate K_bend
        K_bend = 0.388 * alpha * (self.curv)**0.84 * Re**(-0.17)
        
        # Calculate pressure drop (using K_bend)
        dp = K_bend * self.N_bend * 0.5 * rho * u**2
        
        # Update pressure for the state
        p  -= dp
        T   = CP.PropsSI('T', 'P', p, 'S', s, self.fluid)
        rho = CP.PropsSI('D', 'P', p, 'S', s, self.fluid)
        h   = CP.PropsSI('H', 'P', p, 'S', s, self.fluid)
        frac= calc_frac(p, h, fluid='Hydrogen')
        
        # Store results
        results = {'T':    np.array([T]), 
                   'p':    np.array([p]),
                   'rho':  np.array([rho]),
                   'h':    np.array([h]),
                   'frac': np.array([frac])
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

        def solve_H2_state(self, states, T_amb, m_dot, PLOT=False):
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
        