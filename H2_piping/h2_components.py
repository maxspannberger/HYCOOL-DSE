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
    
def area(d):
    return np.pi * d**2 / 4

path = root / "Propulsion" / "only_sizing_results.json"
with open(path, 'r') as file:
    sizes = json.load(file)

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
def get_input_states(states, system, i, m_dot, fluid):
    T   = states['T'][-1][-1]
    p   = states['p'][-1][-1]
    h   = states['h'][-1][-1]
    rho = states['rho'][-1][-1]
    u   = states['u'][-1][-1]
    
    # Get the area of the previous component
    try:
        A_previous_comp = system[i-1].A
    except:
        A_previous_comp = system[i-2].A
       
    A_current_component = system[i].A
    # Perform isentropic expansion if there is an area change
    if A_current_component != A_previous_comp:
        T, p, h, rho, u, _ = isentropic_expansion(p, h, u, m_dot, A_current_component, fluid)
    
    return T, p, h, rho, u



    
# =============================================================================
# Set up a function that can be used in an iterative solver. It uses the conservation
# equations and takes an initial guess
# =============================================================================
def update_states(p1, h1, u1, m_dot, A, fluid, q=0, dp=0, penalty=1e9, w=0):
    
    # Residual function to be used in scipy root solver
    def residuals(vars, p1, h1, u1, m_dot, A, fluid, q=0, dp_fric=0, penalty=1e9):
        rho1 = CP.PropsSI('D', 'P', p1, 'H', h1, fluid)
        
        p2, h2 = vars
        
        try:
            rho2 = CP.PropsSI('D', 'P', p2, 'H', h2, fluid)
            u2   = m_dot / (rho2 * A)
        except ValueError: 
            return [penalty, penalty]
        
        res_momentum   = (p2 + rho2 * u2**2) - (p1 + rho1 * u1**2) + dp_fric
        res_energy     = (h2 + 0.5 * u2**2) - (h1 + 0.5 * u1**2 + q - w)
        
        return [res_momentum, res_energy]
    
    sol = cp_root(residuals,
                  x0=[p1, h1],
                  method='lm',
                  options={'xtol': tol, 'ftol': tol},
                  args=(p1, h1, u1, m_dot, A, fluid, q, dp, config.divergence_penalty))
    p2, h2 = sol.x

    T2    = CP.PropsSI('T', 'P', p2, 'H',  h2, fluid)
    rho2  = CP.PropsSI('D', 'P', p2, 'H',  h2, fluid)
    u2    = m_dot / (rho2 * A)
    frac2 = calc_frac(p2, h2, fluid=fluid)
    
    return T2, p2, h2, rho2, u2, frac2

# =============================================================================
# Set up a function that iteratively solves for the state changes if the flow
# is expanded isentropically
# =============================================================================
def isentropic_expansion(p1, h1, u1, m_dot, A2, fluid, penalty=config.divergence_penalty):
    s = CP.PropsSI('S', 'P', p1, 'H', h1, fluid)
    H = h1 + 0.5*u1**2
    
    # Residual function to be used in scipy root solver
    def residual(vars, s, H, u1, A2, fluid, penalty=config.divergence_penalty):
        p2 = vars[0]
        
        try:
            h2   = CP.PropsSI('H', 'P', p2, 'S',  s, fluid)
            rho2 = CP.PropsSI('D', 'P', p2, 'H', h2, fluid)     
            u2   = m_dot / (rho2 * A2)
            
            return (h2 + 0.5 * u2**2) - H
        except ValueError:
            return penalty 
    
    # Solve for the converge value for p2 and update the other states
    sol = cp_root(residual,
                  x0=[p1],
                  method='lm',
                  options={'xtol': tol, 'ftol': tol},
                  args=(s, H, u1, A2, fluid))
    p2    = sol.x[0]
    h2    = CP.PropsSI('H', 'P', p2, 'S',   s, fluid)
    T2    = CP.PropsSI('T', 'P', p2, 'H',  h2, fluid)
    rho2  = CP.PropsSI('D', 'P', p2, 'H',  h2, fluid)
    u2    = m_dot / (rho2 * A2)
    frac2 = calc_frac(p2, h2, fluid=fluid)
    
    return T2, p2, h2, rho2, u2, frac2


def heat_transfer_coefficient(T1, T2, T_comp, p1, p2, m_dot, d, fluid):
    
    Tf = 0.5 * (0.5 * (T1 + T2) + T_comp)
    pf = 0.5 * (p1 + p2)

    muf = CP.PropsSI('V', 'P', pf, 'T', Tf, fluid)

    Prf = CP.PropsSI('Prandtl', 'P', pf, 'T', Tf, fluid) # Prandtl number
    Ref = 4 * m_dot / (np.pi * d * muf)  # Reynolds number
    # kf = 9.248 + 0.01571 * Tf # thermal conductivity of stainless steel 613L - NOT this one should be used!!!
    kf = CP.PropsSI('L', 'P', pf, 'T', Tf, fluid)

    U = 0.021 * Ref**0.8 * Prf**0.4 * kf / d

    if not Ref >= 10000:
        raise Warning("Formulas used are not valid for the required Reynolds number\n" +\
                "Required Re range: Re >= 1000\n" +\
                f"Used Re: {Ref}"
            )
    if not 0.6 <= Prf <= 160:
        raise Warning("Formulas used are not valid for the required Prandtl number\n" +\
                "Required Pr range: 0.6 <= Pr <= 160\n" +\
                f"Used Pr: {Prf}"
            )

    return U


# =============================================================================
# Iterates through the defined system components to calculate fluid states.
# Updates mass flow rate when splits or merges occur.
# =============================================================================
def solve_system(system, m_dot, T_amb, input_states=None, initial_conditions=None):
    
    if input_states == None:
        states = {'p'   : [],
                  'T'   : [],
                  'rho' : [],
                  'h'   : [],
                  'u'   : [],
                  'frac': []}
    else: 
        states = {'p'   : [np.array([input_states['p'][-1][-1]])],
                  'T'   : [np.array([input_states['T'][-1][-1]])],
                  'rho' : [np.array([input_states['rho'][-1][-1]])],
                  'h'   : [np.array([input_states['h'][-1][-1]])],
                  'u'   : [np.array([input_states['u'][-1][-1]])],
                  'frac': [np.array([input_states['frac'][-1][-1]])]}
        
    HEX_areas = {}
    Temps = {}
    
    for i, comp in enumerate(system):
        # Update the m_dot based on pipe splits and merges
        if type(comp) == tuple:
            m_dot = m_dot * comp[1] / comp[-1]
        else:
            # Propagate the state through the specific component solver
            component_result = comp.solve_H2_state(states, T_amb, m_dot, PLOT=False, system=system, i=i, initial_conditions=initial_conditions)
            initial_conditions = None # just a quick patch, better fix later
            
            states['p'].append(component_result['p'])
            states['T'].append(component_result['T'])
            states['rho'].append(component_result['rho'])
            states['h'].append(component_result['h'])
            states['u'].append(component_result['u'])
            states['frac'].append(component_result['frac'])

            if "area" in component_result:
                 if comp.name not in HEX_areas:
                     HEX_areas[comp.name] = {}
                     Temps[comp.name] = {}
                 HEX_areas[comp.name][comp.location] = {}
                 HEX_areas[comp.name][comp.location]['area'] = component_result['area']
                 HEX_areas[comp.name][comp.location]['pipe_length'] = component_result['pipe_length']
                 HEX_areas[comp.name][comp.location]['N_corners'] = component_result['N_corners']
                 Temps[comp.name][comp.location] = component_result['temperature']

    return states, m_dot, HEX_areas, Temps
    

# =============================================================================
# Define a class for the tank 
# =============================================================================
class Tank:
    def __init__(self):
        self.name = 'Tank'
        self.d = config.tank_d
        self.A = area(self.d)
        self.fluid = config.fluid
        
        self.p   = config.tank_p
        self.T   = config.tank_T
        self.rho = CP.PropsSI('D', 'P', self.p, 'T', self.T, self.fluid)
        self.h   = CP.PropsSI('H', 'P', self.p, 'T', self.T, self.fluid)
        self.frac= calc_frac(self.p, self.h, fluid=self.fluid)
    
    # Function that can be called to calculate the evolution of the state variables
    # in the component
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None, initial_conditions=None):
        
        # Account for isentropic expansion as hydrogen exits the pipe
        u1 = config.tank_initial_u
        
        T2, p2, h2, rho2, u2, frac2 = isentropic_expansion(self.p, self.h, u1, m_dot, self.A, self.fluid)
        
        if frac2 > config.tank_max_gas_frac:
            raise ValueError(f"The hydrogen turns partially gasseous ({frac2}) as it leaves "
                             "the tank. Incompressability assumption doesn't hold.")
        
        # Store the results as a dictionary and return
        results = {'T':   np.array([self.T, T2]), 
                   'p':   np.array([self.p, p2]),
                   'rho': np.array([self.rho, rho2]),
                   'h':   np.array([self.h, h2]),
                   'u':   np.array([u1, u2]),
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
        self.A = area(self.d)            
        self.efficiency = config.pump_efficiency
        self.electric_efficiency = config.pump_electric_efficiency
        self.fluid = config.fluid
        self.name = name

    # Function that can be called to calculate the evolution of the state variables
    # in the component
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None, initial_conditions=None):
        T1, p1, h1, rho1, u1 = get_input_states(states, system, i, m_dot, self.fluid)
        
        s1 = CP.PropsSI('S', 'P', p1, 'H', h1, self.fluid)
        
        try:
            h2_ideal = CP.PropsSI('H', 'P', self.target_p, 'S', s1, self.fluid)
            rho2_ideal = CP.PropsSI('D', 'P', self.target_p, 'S', s1, self.fluid)
        except ValueError:
            raise ValueError(f"Pump failed: Target pressure {self.target_p} Pa is invalid.")
            
        u2_ideal = m_dot / (rho2_ideal * self.A)
        
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
            
            u2_guess = m_dot / (rho2_guess * self.A)
            
            return [(h2_guess + 0.5 * u2_guess**2) - Target_Energy]
        
        h2_sol = cp_root(solve_pump_energy, 
                         x0=[h2_ideal], 
                         method='lm',
                         options={'xtol': tol, 'ftol': tol})
        h2 = h2_sol.x[0]
        
        T2   = CP.PropsSI('T', 'P', p2, 'H', h2, self.fluid)
        rho2 = CP.PropsSI('D', 'P', p2, 'H', h2, self.fluid)
        frac2= calc_frac(p2, h2, fluid=self.fluid)
        u2   = m_dot / (self.A * rho2)
        
        
        power_W = m_dot * w_real / self.electric_efficiency
        print(f"[{self.name}] Pumping to {p2/100000:.1f} bar. (Power per pump: {power_W/1000:.2f} kW)")
        
        results = {'T':    np.array([T2]), 
                   'p':    np.array([p2]),
                   'rho':  np.array([rho2]),
                   'h':    np.array([h2]),
                   'u':    np.array([u2]),
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
                       eps_pipe: float = None,
                       q_set:    float = None):

        self.name      = 'Pipe'
        self.fluid     = config.fluid
        self.length    = length
        
        # Set default pipe parameters if none are overwritten
        self.d         = diameter if diameter is not None else config.pipe_default_d
        self.A         = area(self.d)
        self.segments  = segments if segments is not None else int(np.ceil(length / config.pipe_segment_length))
        self.N         = N        if N is not None        else config.pipe_default_N
        self.N_bar     = N_bar    if N_bar is not None    else config.pipe_default_N_bar
        self.P_mli     = P_mli    if P_mli is not None    else config.pipe_default_P_mli
        self.eps_pipe  = eps_pipe if eps_pipe is not None else config.pipe_default_eps

        self.cs        = config.pipe_mli_cs
        self.cr        = config.pipe_mli_cr
        self.cg        = config.pipe_mli_cg
        self.eps       = config.pipe_mli_eps

        self.q_set     = q_set
    
    # Function that can be called to calculate the evolution of the state variables
    # in the component
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None, initial_conditions=None):
        
        if initial_conditions is None:
            T1, p1, h1, rho1, u1 = get_input_states(states, system, i, m_dot, self.fluid)
        else:
            T1, p1, h1, rho1, u1 = initial_conditions
        
        
        # Get pipe segment dimensions
        dz    = self.length / self.segments
        A_seg = np.pi * self.d * dz
        
        # Initialise results dictionary
        results = {'T':   np.zeros(self.segments), 
                   'p':   np.zeros(self.segments),
                   'rho': np.zeros(self.segments),
                   'h':   np.zeros(self.segments),
                   'u':   np.zeros(self.segments),
                   'frac':np.zeros(self.segments)}

        # Loop over pipe elements to calculate state variable evolution
        for seg in range(self.segments):
            
            mu1  = CP.PropsSI('V', 'P', p1, 'H', h1, self.fluid)
            Re1 = 4 * m_dot / (np.pi * self.d * mu1)

            # --- COMPRESSIBILITY CHECK ---
            # =================================================================
            a_sound = CP.PropsSI('A', 'P', p1, 'H', h1, self.fluid)
            mach_pipe = u1 / a_sound
            
            if mach_pipe >= 1.0:
                raise ValueError(f"[{self.name}] CHOKED FLOW! Mach number {mach_pipe:.3f} >= 1.0 at seg {seg}")
            elif mach_pipe > 0.3:
                print(f"[{self.name}] WARNING: Mach number is {mach_pipe:.2f} at seg {seg}.")
            # =================================================================
            
            # Convert to parameter names as used in the formula
            T_h = T_amb
            T_c = T1
            T_m = (T_h + T_c) / 2
            
            if self.q_set is None:
                Q_dot = A_seg * (
                    (self.cs * T_m * self.N_bar**2.63 * (T_h - T_c)) / (self.N - 1)
                + (self.cr * self.eps * (T_h**4.67 - T_c**4.67)) / self.N
                + (self.cg * (self.P_mli) * (T_h**0.52 - T_c**0.52)) / self.N
                )        
                q       = Q_dot / m_dot
            else:
                q = self.q_set / self.segments

            if Re1 < 2300:
                f = 64 / Re1
            else:
                f = (1 / (-1.8 * np.log10(((self.eps_pipe / self.d) / 3.7)**1.11 + 6.9 / Re1)))**2

            dp_fric = f * (dz / self.d) * 0.5 * rho1 * u1**2
            
            T2, p2, h2, rho2, u2, frac2 = update_states(p1, h1, u1, m_dot, self.A, self.fluid, q=q, dp=dp_fric)          
            
            results['T'][seg]   = T2
            results['p'][seg]   = p2
            results['rho'][seg] = rho2
            results['h'][seg]   = h2
            results['u'][seg]   = u2
            results['frac'][seg]= frac2
            
            T1 = T2
            p1 = p2
            rho1 = rho2
            h1 = h2
            u1 = u2
        
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
        self.A = area(self.d)
        self.fluid = config.fluid
        self.name = name
    
    # Function that can be called to calculate the evolution of the state variables
    # in the component
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None, initial_conditions=None):
        
        T1, p1, h1, rho1, u1 = get_input_states(states, system, i, m_dot, self.fluid)
        
        mu1  = CP.PropsSI('V', 'P', p1, 'H', h1, self.fluid)
        Re1  = 4 * m_dot / (np.pi * self.d * mu1)
        
        alpha = 0.95 + 4.42 * (self.curv)**(-1.96)
        
        K_bend = 0.388 * alpha * (self.curv)**0.84 * Re1**(-0.17)
        
        dp_fric = K_bend * self.N_bend * 0.5 * rho1 * u1**2
        q       = 0
        
        T2, p2, h2, rho2, u2, frac2 = update_states(p1, h1, u1, m_dot, self.A, self.fluid, q=q, dp=dp_fric)
        
        results = {'T':    np.array([T2]), 
                   'p':    np.array([p2]),
                   'rho':  np.array([rho2]),
                   'h':    np.array([h2]),
                   'u':    np.array([u2]),
                   'frac': np.array([frac2])
                   }
        
        return results

# =============================================================================
# Define a class for the components to be cooled
# =============================================================================
class COOL:
    def __init__(self, name:        str, 
                       location:    str,
                       phase:       str,
                       diameter:    float =  config.HEX_default_d,
                       areas:       dict  = None
                       ):   
        
        if "hts" in name:
            self.d = config.HTS_default_d
        else:
            self.d        = diameter
        self.A        = area(self.d)
        self.location = location
        self.name     = name
        self.fluid    = config.fluid
        self.Q_dot    = comps[phase][name][location] * 1000
        self.size     = sizes[name][location]
        self.T = config.operating_temp[self.name]

        if "hts" in self.name:
            self.length = self.size[0]
            self.width = np.pi * self.size[1] # this is PI * D
            self.N_channels = config.HTS_channels
        else:
            self.length = self.size[1]
            self.width = self.size[0]
            self.N_channels = 1

        if areas is None:
            self.area_calc_mode = True
            self.area = None
            self.L = self.length
            self.N_corners = 0
        else:
            self.area_calc_mode = False
            self.area = areas[name][location]['area']
            self.L = areas[name][location]['pipe_length']
            self.N_corners = areas[name][location]['N_corners']

    # Function that can be called to calculate the evolution of the state variables
    # in the component
    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None, initial_conditions=None):
        T1, p1, h1, rho1, u1 = get_input_states(states, system, i, m_dot, self.fluid)
        
        # ---------------------------------------------------------
        # 1. MACRO SYSTEM GEOMETRY (The pipes entering/exiting the component)
        # ---------------------------------------------------------
        # We assume the component connects to the standard system pipe.
        
        # Specific heat added (Total heat / branch mass flow)
        q = self.Q_dot / m_dot 
        q += 34.79 / m_dot  # Heat from fittings and cable extraction divided into all components

        q /= self.N_channels
        m_dot /= self.N_channels

        L_old = np.inf
        j = 0
        while abs((self.L - L_old)/self.L) > 1e-4 and j < 100:
            L_old = self.L
            j += 1

            cumulative_length = 0.0
            internal_system = []

            self.N_corners = int(np.floor(self.L/self.length))
            if self.N_corners > 0:
                R = self.width / (2 * self.N_corners)
                curvature = R / self.d
            else:
                curvature = 2.5

            q_L = q / self.L

            # get internal geometry to compute final states
            while cumulative_length < self.L:
                cumulative_length += self.length

                if cumulative_length >= self.L:
                    remaining_length = self.length - (cumulative_length - self.L)
                    internal_system.extend([Pipe(length=remaining_length, diameter=self.d, q_set=q_L*remaining_length)])
                else:
                    internal_system.extend([
                        Pipe(length=self.length, diameter=self.d, q_set=q_L*self.length),
                        Corner(curv=curvature, diameter=self.d),
                        Corner(curv=curvature, diameter=self.d)
                    ])
            
            solved_internal_system = solve_system(internal_system, m_dot, T_amb, input_states=states,
                                                initial_conditions=(T1, p1, h1, rho1, u1))[0]
            p2 = solved_internal_system['p'][-1][-1]
            T2 = solved_internal_system['T'][-1][-1]
            rho2 = solved_internal_system['rho'][-1][-1]
            h2 = solved_internal_system['h'][-1][-1]
            u2 = solved_internal_system['u'][-1][-1]
            frac2 = solved_internal_system['frac'][-1][-1]

            if "hts" in self.name:
                HEX_effectiveness = config.HEX_effectiveness
            else:
                t = self.d + 2 * config.HEX_extra_thickness
                w_c = self.width * self.length / self.L # + t/2 # the t/2 is added only if the wing tip contributes
                m = np.sqrt(config.h_TMI / (config.k_Al * t))
                HEX_effectiveness = np.tanh(m * w_c) / (m * w_c)

            # print(f"Effectiveness: {HEX_effectiveness}")

            # HEX design
            # f is the "film" temperature (boundary layer of H2 next to the pipe walls)
            if self.area_calc_mode:
                U = heat_transfer_coefficient(T1, T2, self.T, p1, p2, m_dot, self.d, self.fluid)
                deltaT = self.T - 0.5 * (T1 + T2)
                self.area = self.N_channels * self.Q_dot / (U * deltaT * HEX_effectiveness)

            else:
                T_old = 0.0
                k = 0
                while abs(self.T - T_old) > 1e-2 and k < 100:
                    T_old = self.T
                    k += 1

                    U = heat_transfer_coefficient(T1, T2, self.T, p1, p2, m_dot, self.d, self.fluid)
                
                    deltaT = self.Q_dot / (U * self.area * HEX_effectiveness)
                    self.T = deltaT + 0.5 * (T1 + T2)

            self.L = self.area / (np.pi * self.d)
            self.L = config.FPI_relaxation * self.L + (1 - config.FPI_relaxation * L_old)
            print(f"{1000*self.L:.2f}")

            if not self.L/self.d >= 10:
                raise Warning("Formulas used are not valid for the required length/diameter ratio\n" +\
                        "Required L/D range: L/D >= 10\n" +\
                        f"Used L/D: {self.L/self.d}"
                    )

        
        if self.area_calc_mode:
            print(f"{self.name}:")
            print(f"Contact area: {self.area}")
            print(f"Pipe length: {self.L}")
            print(f"Number of corners: {self.N_corners}\n")
        else:
            print(f"{self.name}:")
            print(f"Temperature: {self.T}\n")


        results = {'T':   np.array([T2]), 
                   'p':   np.array([p2]),
                   'rho': np.array([rho2]),
                   'h':   np.array([h2]),
                   'u':   np.array([u2]),
                   'frac':np.array([frac2]),
                   'area': self.area,
                   'pipe_length': self.L,
                   'N_corners': self.N_corners,
                   'temperature': self.T}
        
        return results
    
# =============================================================================
# Define a class for the valves
# =============================================================================
class Valve:
    def __init__(self, name:        str,
                       diameter:    float =  config.pipe_default_d,
                       ):

        self.name     = name
        self.fluid    = config.fluid
        self.d        = diameter
        self.A        = area(self.d)

    def solve_H2_state(self, states, T_amb, m_dot, system, PLOT=False, i=None, initial_conditions=None):
        
        T1, p1, h1, rho1, u1 = get_input_states(states, system, i, m_dot, self.fluid)

        # --- COMPRESSIBILITY CHECK ---
        # =====================================================================
        a_sound = CP.PropsSI('A', 'P', p1, 'H', h1, self.fluid)
        mach_valve = u1 / a_sound
        if mach_valve >= 1.0:
            raise ValueError(f"[{self.name} valve] CHOKED FLOW! Mach number {mach_valve:.3f} >= 1.0")
        elif mach_valve > 0.3:
            print(f"[{self.name} valve] WARNING: Mach number is {mach_valve:.2f}. Compressibility high.")
        # =====================================================================

        Q   = (m_dot / rho1) * 15850.3         # Q in gallons per minute
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
        
        T2, p2, h2, rho2, u2, frac2 = update_states(p1, h1, u1, m_dot, self.A, self.fluid, q=0, dp=dp)
        
        results = {'T':    np.array([T2]), 
                   'p':    np.array([p2]),
                   'rho':  np.array([rho2]),
                   'h':    np.array([h2]),
                   'u':    np.array([u2]),
                   'frac': np.array([frac2])
                   }
        
        return results
    
