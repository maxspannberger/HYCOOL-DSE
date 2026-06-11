from pathlib import Path
import sys
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from scipy.optimize import root as cp_root
import json

from system_config import H2SystemConfig 

config = H2SystemConfig()
tol = config.max_error

path = root / "Propulsion" / "only_cooling_results.json"
with open(path, 'r') as file:
    comps = json.load(file)

# =============================================================================
# Calculate the fraction of gas. Get rid of supercriticals by forcing to o or 1
# =============================================================================
def calc_frac(p, h, fluid='Hydrogen'):
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
# Get the input states by taking last stored state values        
# =============================================================================
def get_input_states(states):
    T   = states['T'][-1][-1]
    p   = states['p'][-1][-1]
    h   = states['h'][-1][-1]
    rho = states['rho'][-1][-1]
    
    return T, p, h, rho

# =============================================================================
# Set up av function that can be used in an iterative solver. It uses the conservation
# equations and takes an initial guess
# =============================================================================
def update_states(vars, p1, h1, u1, m_dot, A_cs, fluid, q=0, dp_fric=0, penalty_val=1e9):
    rho1 = CP.PropsSI('D', 'P', p1, 'H', h1, fluid)
    
    p2, h2 = vars
    
    try:
        rho2 = CP.PropsSI('D', 'P', p2, 'H', h2, fluid)
        u2   = m_dot / (rho2 * A_cs)
    except ValueError: 
        return [penalty_val, penalty_val]
    
    res_momentum   = (p2 + rho2 * u2**2) - (p1 + rho1 * u1**2) + dp_fric
    res_energy     = (h2 + 0.5 * u2**2) - (h1 + 0.5 * u1**2 + q)
    
    return [res_momentum, res_energy]

# =============================================================================
# Define a class for the tank 
# =============================================================================
class Tank:
    def __init__(self):
        self.name = 'Tank'
        self.d = config.tank_d
        self.fluid = config.fluid
        
        self.p   = config.tank_p
        self.T   = config.tank_T
        self.rho = CP.PropsSI('D', 'P', self.p, 'T', self.T, self.fluid)
        self.h   = CP.PropsSI('H', 'P', self.p, 'T', self.T, self.fluid)
        self.frac= calc_frac(self.p, self.h, fluid=self.fluid)
    
    # Function that can be called to calculate the evolution of the state variables
    # in the component
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None):
        A_cs = np.pi * self.d**2 / 4
        u    = config.tank_initial_u
        
        # Iteratively solve for the upstream states
        sol = cp_root(update_states,
                      x0=[self.p * 0.999, self.h],
                      method='lm',
                      args=(self.p, self.h, u, m_dot, A_cs, self.fluid, 0, 0, config.divergence_penalty))
        p2, h2 = sol.x
        
        # Update the remaining state variables based on h and p
        rho2  = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        T2    = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        frac2 = calc_frac(p2, h2, fluid=self.fluid)
        if frac2 > config.tank_max_gas_frac:
            raise ValueError(f"The hydrogen turns partially gasseous ({frac2}) as it leaves "
                             "the tank. Incompressability assumption doesn't hold.")
        
        # Store the results as a dictionary and return
        results = {'T':   np.array([self.T, T2]), 
                   'p':   np.array([self.p, p2]),
                   'rho': np.array([self.rho, rho2]),
                   'h':   np.array([self.h, h2]),
                   'frac':np.array([self.frac, frac2])
                   }
        
        return results

# =============================================================================
# Define a class for the pump
# =============================================================================
class Pump:
    def __init__(self, target_p: float,
                       diameter: float,
                       name: str = 'CryoPump'):
        
        self.target_p = target_p      
        self.d = diameter              
        self.efficiency = config.pump_efficiency
        self.electric_efficiency = config.pump_electric_efficiency
        self.fluid = config.fluid
        self.name = name

    # Function that can be called to calculate the evolution of the state variables
    # in the component
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None):
        T1, p1, h1, rho1 = get_input_states(states)
        
        A_cs = np.pi * self.d**2 / 4 
        
        u1 = m_dot / (rho1 * A_cs)
        
        s1 = CP.PropsSI('S', 'P', p1, 'H', h1, self.fluid)
        
        try:
            h2_ideal = CP.PropsSI('H', 'P', self.target_p, 'S', s1, self.fluid)
            rho2_ideal = CP.PropsSI('D', 'P', self.target_p, 'S', s1, self.fluid)
        except ValueError:
            raise ValueError(f"Pump failed: Target pressure {self.target_p} Pa is invalid.")
            
        u2_ideal = m_dot / (rho2_ideal * A_cs)
        
        w_ideal = (h2_ideal + 0.5 * u2_ideal**2) - (h1 + 0.5 * u1**2)
        
        w_real = w_ideal / self.efficiency
        
        Target_Energy = (h1 + 0.5 * u1**2) + w_real
        p2 = self.target_p
        
        def solve_pump_energy(vars):
            h2_guess = vars[0]
            
            try:
                rho2_guess = CP.PropsSI('D', 'P', p2, 'H', h2_guess, self.fluid)
            except ValueError:
                return [config.divergence_penalty] 
            
            u2_guess = m_dot / (rho2_guess * A_cs)
            
            return [(h2_guess + 0.5 * u2_guess**2) - Target_Energy]
        
        h2_sol = cp_root(solve_pump_energy, 
                         x0=[h2_ideal], 
                         method='lm',
                         options={'xtol': tol, 'ftol': tol})
        h2 = h2_sol.x[0]
        
        T2   = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        rho2 = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        frac2= calc_frac(p2, h2, fluid=self.fluid)
        
        power_W = m_dot * w_real / self.electric_efficiency
        print(f"[{self.name}] Pumping to {p2/100000:.1f} bar. (Power per pump: {power_W/1000:.2f} kW)")
        
        results = {'T':    np.array([T2]), 
                   'p':    np.array([p2]),
                   'rho':  np.array([rho2]),
                   'h':    np.array([h2]),
                   'frac': np.array([frac2])
                   }
        
        return results
    
# =============================================================================
#  Define a class for the pipe       
# =============================================================================
class Pipe:
    def __init__(self, length:   float,
                       diameter: float = None,   
                       segments: int   = None,     
                       N:        int   = None,
                       N_bar:    float = None,      
                       P_mli:    float = None,
                       eps_pipe: float = None):

        self.name      = 'Pipe'
        self.fluid     = config.fluid
        self.length    = length
        
        # Set default pipe parameters if none are overwritten
        self.d         = diameter if diameter is not None else config.pipe_default_d
        self.segments  = segments if segments is not None else int(length / config.pipe_segment_length)
        self.N         = N        if N is not None        else config.pipe_default_N
        self.N_bar     = N_bar    if N_bar is not None    else config.pipe_default_N_bar
        self.P_mli     = P_mli    if P_mli is not None    else config.pipe_default_P_mli
        self.eps_pipe  = eps_pipe if eps_pipe is not None else config.pipe_default_eps

        self.cs        = config.pipe_mli_cs
        self.cr        = config.pipe_mli_cr
        self.cg        = config.pipe_mli_cg
        self.eps       = config.pipe_mli_eps
    
    # Function that can be called to calculate the evolution of the state variables
    # in the component
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None):
        
        T1, p1, h1, rho1 = get_input_states(states)
        
        results = {'T':   np.zeros(self.segments), 
                   'p':   np.zeros(self.segments),
                   'rho': np.zeros(self.segments),
                   'h':   np.zeros(self.segments),
                   'frac':np.zeros(self.segments)}
        
        dz    = self.length / self.segments
        A_cs  = np.pi * self.d**2 / 4
        A_seg = np.pi * self.d * dz

        # Loop over pipe elements to calculate state variable evolution
        for seg in range(self.segments):
            mu1  = CP.PropsSI('V', 'P', p1, 'H', h1, self.fluid)

            u1  = m_dot / (rho1 * A_cs)
            Re1 = 4 * m_dot / (np.pi * self.d * mu1)

            # --- COMPRESSIBILITY CHECK ---
            a_sound = CP.PropsSI('A', 'P', p1, 'H', h1, self.fluid)
            mach_pipe = u1 / a_sound
            
            if mach_pipe >= 1.0:
                raise ValueError(f"[{self.name}] CHOKED FLOW! Mach number {mach_pipe:.3f} >= 1.0 at seg {seg}")
            elif mach_pipe > 0.3:
                print(f"[{self.name}] WARNING: Mach number is {mach_pipe:.2f} at seg {seg}.")
            # -----------------------------
            
            # Convert to parameter names as used in the formula
            T_h = T_amb
            T_c = T1
            T_m = (T_h + T_c) / 2
            
            Q_dot = A_seg * (
                (self.cs * T_m * self.N_bar**2.63 * (T_h - T_c)) / (self.N - 1)
              + (self.cr * self.eps * (T_h**4.67 - T_c**4.67)) / self.N
              + (self.cg * (self.P_mli) * (T_h**0.52 - T_c**0.52)) / self.N
            )
            
            if Re1 < 2300:
                f = 64 / Re1
            else:
                f = (1 / (-1.8 * np.log10(((self.eps_pipe / self.d) / 3.7)**1.11 + 6.9 / Re1)))**2
            
            q       = Q_dot / m_dot
            dp_fric = f * (dz / self.d) * 0.5 * rho1 * u1**2
            
            sol = cp_root(update_states,
                          x0=[p1, h1],
                          method='lm',
                          options={'xtol': tol, 'ftol': tol},
                          args=(p1, h1, u1, m_dot, A_cs, self.fluid, q, dp_fric, config.divergence_penalty))
            p2, h2 = sol.x
            
            T2     = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
            rho2   = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
            frac2  = calc_frac(p2, h2, fluid=self.fluid)            
            
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
            
            plot_data = [
                (results['T'], 'Temperature (K)', 'tab:red'),
                (results['p'], 'Pressure (Pa)', 'tab:blue'),
                (results['rho'], 'Density (kg/m³)', 'tab:green'),
                (results['h'], 'Enthalpy (J/kg)', 'tab:purple')
            ]
            
            x_values = np.linspace(0, self.length, self.segments)
            
            for idx, (data, label, color) in enumerate(plot_data):
                ax[idx].plot(x_values, data, color=color, linewidth=2)
                ax[idx].set_ylabel(label)
                ax[idx].grid(True, linestyle='--', alpha=0.5)
            
            fig.suptitle('H2 State Across Pipe', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            
        return results
    
# =============================================================================
#  Define a class for the corners   
# =============================================================================
class Corner:
    def __init__(self, curv:     float,
                       diameter: float   =  config.pipe_default_d,
                       name:      str    = 'Bend',
                       N_bend:    int    = 1
                       ):
        
        self.N_bend = N_bend
        self.curv = curv
        self.d = diameter
        self.fluid = config.fluid
        self.name = name
    
    # Function that can be called to calculate the evolution of the state variables
    # in the component
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None):
        T1, p1, h1, rho1 = get_input_states(states)
        
        mu1  = CP.PropsSI('V', 'P', p1, 'H', h1, self.fluid)
        Re1  = 4 * m_dot / (np.pi * self.d * mu1)
        
        A_cs = np.pi * self.d**2 / 4
        u1 = m_dot / (rho1 * A_cs)
        
        alpha = 0.95 + 4.42 * (self.curv)**(-1.96)
        
        K_bend = 0.388 * alpha * (self.curv)**0.84 * Re1**(-0.17)
        
        dp_fric = K_bend * self.N_bend * 0.5 * rho1 * u1**2
        q       = 0
        
        sol = cp_root(update_states,
                      x0=[p1, h1],
                      method='lm',
                      options={'xtol': tol, 'ftol': tol},
                      args=(p1, h1, u1, m_dot, A_cs, self.fluid, q, dp_fric, config.divergence_penalty))
        p2, h2 = sol.x
        
        T2    = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        rho2  = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        frac2 = calc_frac(p2, h2, fluid=self.fluid)
        
        results = {'T':    np.array([T2]), 
                   'p':    np.array([p2]),
                   'rho':  np.array([rho2]),
                   'h':    np.array([h2]),
                   'frac': np.array([frac2])
                   }
        
        return results

# =============================================================================
# Define a class for the components to be cooled
# =============================================================================
class COOL:
    def __init__(self, name:        str, 
                       location:    str,
                       phase:       str   =  config.phase
                       ):   
        
        self.location = location
        self.name     = name
        self.fluid    = config.fluid
        self.Q_dot    = comps[phase][name][location] * 1000

    # Function that can be called to calculate the evolution of the state variables
    # in the component
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None):
        T1, p1, h1, rho1 = get_input_states(states)
        
        # ---------------------------------------------------------
        # 1. MACRO SYSTEM GEOMETRY (The pipes entering/exiting the component)
        # ---------------------------------------------------------
        # We assume the component connects to the standard system pipe.
        d_pipe = config.pipe_default_d  
        A_pipe = np.pi * d_pipe**2 / 4
        T_component = config.operating_temp[self.name]
        
        # Macro inlet velocity from the upstream pipe
        u1 = m_dot / (rho1 * A_pipe)
        
        # Specific heat added (Total heat / branch mass flow)
        q = self.Q_dot / m_dot 

        # ---------------------------------------------------------
        # 2. MICRO INTERNAL GEOMETRY (Calculating the friction drop)
        # ---------------------------------------------------------
        if self.name in ['hts_gen', 'hts_pow']:
            eps_hts = config.eps_hts
            N_slots = config.N_slots
            A_slot_tot = config.A_slot * 6  # Total geometric stator area 
            L = config.L
            VF = config.VF

            # Micro geometry for a single cooling slot
            A_slot = A_slot_tot / N_slots
            A_flow_slot = A_slot * VF 
            m_dot_slot = m_dot / N_slots

            P_wet = 2*np.pi*np.sqrt((1-VF)*A_slot/np.pi) + 2*(0.0318 + 0.0484)
            Dh = 4 * A_flow_slot / P_wet 

            mu1 = CP.PropsSI('V', 'P', p1, 'H', h1, self.fluid)

            # INTERNAL velocity (Used strictly for friction, NOT for the macro energy balance!)
            u_internal = m_dot_slot / (rho1 * A_flow_slot) 
            Re_internal = 4 * m_dot_slot / (np.pi * Dh * mu1)
        
            if Re_internal < 2300:
                f = 64 / Re_internal
            else:
                f = (1 / (-1.8 * np.log10(((eps_hts / Dh) / 3.7)**1.11 + 6.9 / Re_internal)))**2
            
            # Pure micro-channel friction
            dp_fric = f * (L/Dh) * (rho1 * u_internal**2 / 2) 
            
            # Optional minor losses for contraction/expansion moving from pipe -> tiny slots -> pipe
            dp_minor = (0.5 + 1.0) * (rho1 * u_internal**2 / 2)
            dp_fric += dp_minor

        else:
            # ---------------------------------------------------------
            # NON-HTS COMPONENTS (AC/DC, Bus, etc.)
            # ---------------------------------------------------------
            # Since we lack cold-plate micro geometry, we use the config dummy pressure drop 
            dp_fric = config.cool_dummy_dp 

        # ---------------------------------------------------------
        # --- COMPRESSIBILITY CHECK ---
        # ---------------------------------------------------------
        a_sound = CP.PropsSI('A', 'P', p1, 'H', h1, self.fluid)
        
        # Macro Pipe Check
        mach_macro = u1 / a_sound
        if mach_macro >= 1.0:
            raise ValueError(f"[{self.name}] CHOKED FLOW! Macro Mach number {mach_macro:.3f} >= 1.0")
        elif mach_macro > 0.3:
            print(f"[{self.name}] WARNING: Macro Mach number is {mach_macro:.2f}. Compressibility high.")

        # Micro Slot Check (Only for HTS)
        if self.name in ['hts_gen', 'hts_pow']:
            mach_micro = u_internal / a_sound
            if mach_micro >= 1.0:
                raise ValueError(f"[{self.name}] CHOKED FLOW IN SLOTS! Micro Mach {mach_micro:.3f} >= 1.0")
            elif mach_micro > 0.3:
                print(f"[{self.name}] WARNING: Micro Mach number in slots is {mach_micro:.2f}.")
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # 3. MACRO SOLVER EXECUTION
        # ---------------------------------------------------------
        # We pass u1 (inlet pipe velocity) and A_pipe (outlet pipe area). 
        # This conserves momentum and kinetic energy correctly across the component jump.
        sol = cp_root(update_states,
                      x0=[p1, h1],
                      method='lm',
                      options={'xtol': tol, 'ftol': tol},
                      args=(p1, h1, u1, m_dot, A_pipe, self.fluid, q, dp_fric, config.divergence_penalty))
        p2, h2 = sol.x
                    
        T2    = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        rho2  = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        frac2 = calc_frac(p2, h2, fluid=self.fluid)


        # HEX design
        # f is the "film" temperature
        Tf = 0.5 * (0.5 * (T1 + T2) + T_component)
        pf = 0.5 * (p1 + p2)
        muf = CP.PropsSI('V', 'P', pf, 'T', Tf, self.fluid)
        D_input = system[i-1].d

        Prf = CP.PropsSI('Prandtl', 'P', pf, 'T', Tf, self.fluid) # Prandtl number
        Ref = 4 * m_dot / (np.pi * D_input * muf)  # Reynolds number
        kf = 9.248 + 0.01571 * Tf # thermal conductivity of stainless steel 613L
        U = 0.021 * Ref**0.8 * Prf**0.4 * kf / D_input
        
        deltaT = T_component - 0.5 * (T1 + T2)
        A_contact = 1000 * self.Q_dot / (U * deltaT)
        pipe_length = A_contact / (np.pi * D_input)

        results = {'T':   np.array([T2]), 
                   'p':   np.array([p2]),
                   'rho': np.array([rho2]),
                   'h':   np.array([h2]),
                   'frac':np.array([frac2]),
                   'A_contact': np.array([A_contact]),
                   'pipe_length': np.array([pipe_length])}
        
        return results
    
# =============================================================================
# Define a class for the valves
# =============================================================================
class Valve:
    def __init__(self, name:        str,
                       diameter:    float =  config.pipe_default_d,
                       phase:       str   =  config.phase
                       ):

        self.name     = name
        self.fluid    = config.fluid
        self.d        = diameter
        self.phase    = phase

    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None):
        T1, p1, h1, rho1 = get_input_states(states)
        A_cs  = np.pi * self.d**2 / 4
        u1 = m_dot / (A_cs * rho1)

        # --- COMPRESSIBILITY CHECK ---
        a_sound = CP.PropsSI('A', 'P', p1, 'H', h1, self.fluid)
        mach_valve = u1 / a_sound
        if mach_valve >= 1.0:
            raise ValueError(f"[{self.name} valve] CHOKED FLOW! Mach number {mach_valve:.3f} >= 1.0")
        elif mach_valve > 0.3:
            print(f"[{self.name} valve] WARNING: Mach number is {mach_valve:.2f}. Compressibility high.")
        # -----------------------------

        Q   = (m_dot / rho1) * 15850.3        # Q in gallons per minute
        S_g = rho1 / 999                       # rho_h2 / rho_water

        if self.name == 'check':
            Cv = 22413 * self.d ** 2.0817
            dp = (S_g / ((Cv / Q) ** 2)) * 6895  # convert from psi to Pa
            print(f"[{self.name} valve] Pressure drop: {dp:.2f} Pa")

        elif self.name == 'shutoff':
            Cv = 1173.6 * self.d - 10.19
            if Cv <= 0:
                raise ValueError(f"[shutoff valve] Non-physical Cv={Cv:.4f} for diameter {self.d*1000:.1f} mm. "
                                  "Minimum valid diameter is ~8.7 mm.")
            dp = (S_g / ((Cv / Q) ** 2)) * 6895  # convert from psi to Pa
            print(f"[{self.name} valve] Pressure drop: {dp:.2f} Pa")
        else:
            raise TypeError('Invalid valve type')
                  
        q  = 0
        
        sol = cp_root(update_states,
                      x0=[p1, h1],
                      method='lm',
                      options={'xtol': tol, 'ftol': tol},
                      args=(p1, h1, u1, m_dot, A_cs, self.fluid, q, dp, config.divergence_penalty))
        p2, h2 = sol.x
        
        T2    = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        rho2  = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        frac2 = calc_frac(p2, h2, fluid=self.fluid)
        
        results = {'T':    np.array([T2]), 
                   'p':    np.array([p2]),
                   'rho':  np.array([rho2]),
                   'h':    np.array([h2]),
                   'frac': np.array([frac2])
                   }
        
        return results
    
