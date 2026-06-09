from pathlib import Path
import sys
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import json

# Import the system configuration class from its separate file
from system_config import H2SystemConfig


config = H2SystemConfig()


# =============================================================================
# In order to track the gas fraction, this function outputs whether the coolprop
# value should be used or a manually picked value should be used 
# (mainly important for supercritical phases)
# =============================================================================
def calc_frac(p, h, fluid='Hydrogen'):
    # Determine the phase
    phase = CP.PhaseSI('P', p, 'H', h, fluid)
    
    if phase == 'twophase':
        return CP.PropsSI('Q', 'P', p, 'H', h, fluid)
    elif phase == 'liquid' or phase == 'supercritical_liquid':
        return 0.0
    elif phase == 'gas' or phase == 'supercritical_gas' or phase == 'supercritical':
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
def update_states(vars, p1, h1, u1, m_dot, A_cs, fluid, q=0, dp_fric=0, penalty_val=1e9):
    rho1 = CP.PropsSI('D', 'P', p1, 'H', h1, fluid)
    
    p2, h2 = vars
    
    try:
        rho2 = CP.PropsSI('D', 'P', p2, 'H', h2, fluid)
        u2   = m_dot / (rho2 * A_cs)
    except: 
        return [penalty_val, penalty_val] # Reset guess values if the solver diverges
    
    res_momentum   = (p2 + rho2 * u2**2) - (p1 + rho1 * u1**2) + dp_fric
    res_energy     = (h2 + 0.5 * u2**2) - (h1 + 0.5 * u1**2 + q)
    
    return [res_momentum, res_energy]


# =============================================================================
# Define the tank class
# =============================================================================
class Tank:
    def __init__(self, diameter: float,
                       p:        float,
                       T:        float):
        
        # Initialise the tank specific parameters
        self.name = 'Tank'
        self.d = diameter
        self.fluid = config.fluid
        
        # The state of the hydrogen in the tank based on the assumption that
        # the fluid is fully in a liquid state
        self.p   = p
        self.T   = T
        self.rho = CP.PropsSI('D', 'P', self.p, 'T|liquid', self.T, self.fluid)
        self.h   = CP.PropsSI('H', 'P', self.p, 'T|liquid', self.T, self.fluid)
        self.frac= calc_frac(self.p, self.h, fluid=self.fluid)
      
    # Set the tank values as the initial values of the piping system
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False):
        A_cs = np.pi * system[1].d**2 / 4
        u    = config.tank_initial_u
        
        # Update pressure and enthalpy
        p2, h2 = fsolve(update_states,
                      x0=[self.p, self.h],
                      args=(self.p, self.h, u, m_dot, A_cs, self.fluid, 0, 0, config.divergence_penalty))
        
        rho2  = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        T2    = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        frac2 = calc_frac(p2, h2, fluid=self.fluid)
        if frac2 > config.tank_max_gas_frac:
            raise ValueError(f"The hydrogen turns partially gasseous ({frac2}) as it leaves "
                             "the tank. Incompressability assumption doesn't hold.")
        
        # Store results in dictionary and return
        results = {'T':   np.array([self.T, T2]), 
                   'p':   np.array([self.p, p2]),
                   'rho': np.array([self.rho, rho2]),
                   'h':   np.array([self.h, h2]),
                   'frac':np.array([self.frac, frac2])
                   }
        
        return results


# =============================================================================
# Define the Cryogenic Pump class
# =============================================================================
class Pump:
    def __init__(self, target_p: float,
                       diameter: float,
                       efficiency: float,
                       name: str = 'CryoPump'):
        
        self.target_p = target_p      
        self.d = diameter             
        self.efficiency = efficiency  
        self.fluid = config.fluid
        self.name = name

    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False):
        # Extract states from the end of the previous component
        T1, p1, h1, rho1 = get_input_states(states)
        
        # Determine the cross-sectional area (ensuring Continuity)
        A_cs = np.pi * self.d**2 / 4 
        
        # Calculate inlet velocity
        u1 = m_dot / (rho1 * A_cs)
        
        # Find inlet entropy to lock the ideal benchmark
        s1 = CP.PropsSI('S', 'P', p1, 'H', h1, self.fluid)
        
        # Calculate the IDEAL (Isentropic) target state
        try:
            h2_ideal = CP.PropsSI('H', 'P', self.target_p, 'S', s1, self.fluid)
            rho2_ideal = CP.PropsSI('D', 'P', self.target_p, 'S', s1, self.fluid)
        except ValueError:
            raise ValueError(f"Pump failed: Target pressure {self.target_p} Pa is invalid.")
            
        u2_ideal = m_dot / (rho2_ideal * A_cs)
        
        # Calculate Ideal Work required (Full First Law Form)
        w_ideal = (h2_ideal + 0.5 * u2_ideal**2) - (h1 + 0.5 * u1**2)
        
        # Calculate REAL Work injected by the inefficient impeller
        w_real = w_ideal / self.efficiency
        
        # Define the exact Total Energy that must leave the pump
        Target_Energy = (h1 + 0.5 * u1**2) + w_real
        p2 = self.target_p
        
        # Rigorous Numerical Solver: Drive the First Law Residual to Zero
        def solve_pump_energy(vars):
            h2_guess = vars[0]
            
            # Request density from CoolProp based on the guessed static enthalpy
            try:
                rho2_guess = CP.PropsSI('D', 'P', p2, 'H', h2_guess, self.fluid)
            except:
                return [config.divergence_penalty] 
            
            # Calculate corresponding velocity
            u2_guess = m_dot / (rho2_guess * A_cs)
            
            # Return the Residual Error: (Proposed Total Energy) - (Target Total Energy)
            return [(h2_guess + 0.5 * u2_guess**2) - Target_Energy]
        
        # Execute the solver, using the ideal enthalpy as the initial guess
        h2_sol = fsolve(solve_pump_energy, [h2_ideal])
        h2 = h2_sol[0]
        
        # Lock in final real state properties
        T2   = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        rho2 = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        frac2= calc_frac(p2, h2, fluid=self.fluid)
        
        # Print power estimation
        power_W = m_dot * w_real
        print(f"[{self.name}] Pumping to {p2/100000:.1f} bar. (Power: {power_W/1000:.2f} kW)")
        
        # Store and return results
        results = {'T':    np.array([T2]), 
                   'p':    np.array([p2]),
                   'rho':  np.array([rho2]),
                   'h':    np.array([h2]),
                   'frac': np.array([frac2])
                   }
        
        return results
    
            
# =============================================================================
# Define the pipe class
# =============================================================================
class Pipe:
    def __init__(self, length:   float,
                       diameter: float = None,   
                       segments: int   = None,     
                       N:        int   = None,
                       N_bar:    float = None,      
                       P_mli:    float = None,
                       eps_pipe: float = None):

        # Initialise the pipe specific parameters
        self.name      = 'Pipe'
        self.fluid     = config.fluid
        self.length    = length
        
        # Pull geometric traits from system_config if they are left unassigned
        self.d         = diameter if diameter is not None else config.pipe_default_d
        self.segments  = segments if segments is not None else config.pipe_default_segments
        self.N         = N        if N is not None        else config.pipe_default_N
        self.N_bar     = N_bar    if N_bar is not None    else config.pipe_default_N_bar
        self.P_mli     = P_mli    if P_mli is not None    else config.pipe_default_P_mli
        self.eps_pipe  = eps_pipe if eps_pipe is not None else config.pipe_default_eps

        # Note: P_mli must be supplied in Pa; converted to Torr internally
        # because the Lockheed C_G constant was fitted with pressure in Torr
        self.cs        = config.pipe_mli_cs
        self.cr        = config.pipe_mli_cr
        self.cg        = config.pipe_mli_cg
        self.eps       = config.pipe_mli_eps
     
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
              + (self.cg * (self.P_mli) * (T_h**0.52 - T_c**0.52)) / self.N
            )
            
            # Determine the Friction factor:
            # either Hagen-Poiseuille (laminar) or Haaland (turbulent)
            if Re1 < 2300:
                f = 64 / Re1
            else:
                f = (1 / (-1.8 * np.log10(((self.eps_pipe / self.d) / 3.7)**1.11 + 6.9 / Re1)))**2
            
            # Update the enthalpy based on the heat leak
            q       = Q_dot / m_dot
            dp_fric = f * (dz / self.d) * 0.5 * rho1 * u1**2
            
            p2, h2 = fsolve(update_states,
                          x0=[p1, h1],
                          args=(p1, h1, u1, m_dot, A_cs, self.fluid, q, dp_fric, config.divergence_penalty))
            
            # Update the state variables
            T2     = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
            rho2   = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
            frac2  = calc_frac(p2, h2, fluid=self.fluid)            
            
            # Store the updated state variables
            results['T'][seg]   = T2
            results['p'][seg]   = p2
            results['rho'][seg] = rho2
            results['h'][seg]   = h2
            results['frac'][seg]= frac2
            
            T1 = T2
            p1 = p2
            rho1 = rho2
            h1 = h2
        
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
    def __init__(self, curv:     float,
                       diameter: float,
                       name:      str    = 'Bend',
                       N_bend:    int    = 1
                       ):
        
        self.N_bend = N_bend
        self.curv = curv
        self.d = diameter
        self.fluid = config.fluid
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
                      args=(p1, h1, u1, m_dot, A_cs, self.fluid, q, dp_fric, config.divergence_penalty))
        
        T2    = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        rho2  = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        frac2 = calc_frac(p2, h2, fluid=self.fluid)
        
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
class COOL:
    def __init__(self, location:    str,
                       name:        str 
                       ):   
        
        self.location = location
        self.name     = name
        self.fluid    = config.fluid

        # Load the component cooling requirements from the propulsion json file
        path = str(root / "Propulsion/component_sizing_results.json")
        with open(path, 'r') as file:
            comps = json.load(file)
        self.Q_dot    = comps[name][location]['P_cool']

    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False):
        # Extract states from end of previous component
        T, p, h, rho = get_input_states(states)
        q = self.Q_dot / m_dot
        A_out = config.cool_dummy_A
        u = config.cool_dummy_u

        dp_fric = config.cool_dummy_dp

        if self.name in ['hts_gen', 'hts_pow']:
            # Constants
            eps_hts = config.eps_hts
            
            # 1. MISTAKE 4 FIX: config.A_slot is ALREADY the total stator area
            A_slot_tot = config.A_slot 
            N_slots = config.N_slots
            L = config.L
            VF = config.VF

            # 2. MISTAKE 3 FIX: Calculate physical area AND actual fluid flow area
            A_slot = A_slot_tot / N_slots
            A_flow_slot = A_slot * VF 
            m_dot_slot = m_dot / N_slots

            # Slot wetted area
            P_wet = 2*np.pi*np.sqrt((1-VF)*A_slot/np.pi) + 2*(0.0318 + 0.0484)

            # Hydraulic diameter (Using FLOW area, not total slot area)
            Dh = 4 * A_flow_slot / P_wet 

            # Fluid properties at segment inlet
            mu1  = CP.PropsSI('V', 'P', p, 'H', h, self.fluid)

            # Flow velocity and Reynolds number (Using FLOW area)
            u_slot = m_dot_slot / (rho * A_flow_slot) 
            Re1 = 4 * m_dot_slot / (np.pi * Dh * mu1)
        
            if Re1 < 2300:
                f = 64 / Re1
            else:
                f = (1 / (-1.8 * np.log10(((eps_hts / Dh) / 3.7)**1.11 + 6.9 / Re1)))**2
            
            # 3. Parallel pressure drop is just the drop of one channel.
            dp_fric = f * (L/Dh) * (rho * u_slot**2 / 2)  

            dp_fric = f * (L/Dh) * (rho * u_slot**2 / 2)
            #print(f"[{self.name}] Computed Friction Drop: {dp_fric:.4f} Pa")
            print(f"[{self.name}] Computed Mass flow rate: {m_dot:.4f} kg/s")
            
            # 4. Set variables for the fsolve args so momentum is conserved properly
            u = u_slot # The inlet velocity
            A_out = A_slot_tot * VF # The total outlet flow area for the momentum balance
        
        p2, h2 = fsolve(update_states,
                      x0=[p, h],
                      args=(p, h, u, m_dot, A_out, self.fluid, q, dp_fric, config.divergence_penalty))
                   
        T2    = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        rho2  = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        frac2 = calc_frac(p2, h2, fluid=self.fluid)
        
        # Store results in dictionary and return
        results = {'T':   np.array([T2]), 
                   'p':   np.array([p2]),
                   'rho': np.array([rho2]),
                   'h':   np.array([h2]),
                   'frac':np.array([frac2])
                   }
        
        return results