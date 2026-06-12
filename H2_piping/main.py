from pathlib import Path
import sys

# Set up paths to ensure we can import local modules
folder = Path(__file__).resolve().parent
sys.path.append(str(folder))

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

# Import your components from h2_components and configuration from system_config
from h2_components import Tank, Pipe, Pump, Corner, COOL, Valve

from rich import print as rich_printv
from rich.tree import Tree
import matplotlib.pyplot as plt
import numpy as np
import json
from system_config import H2SystemConfig
c = H2SystemConfig()  

# =============================================================================
# Save final states to JSON
# =============================================================================
def save_results_to_json(phase_name, T, p, rho, h, m_dot_final):
    results_file = root / "Propulsion" / "final_states.json"
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
        
    data[phase_name] = {
        "Temperature_K": round(float(T), 2),
        "Pressure_Pa": round(float(p), 2),
        "Density_kg_m3": round(float(rho), 2),
        "Enthalpy_J_kg": round(float(h), 2),
        "Final_MassFlow_kg_s": round(float(m_dot_final), 5)
    }
    
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=4)

# =============================================================================
# Iterates through the defined system components to calculate fluid states.
# Updates mass flow rate when splits or merges occur.
# =============================================================================
def solve_system(system, m_dot, T_amb):
   
    states = {'p'   : [],
              'T'   : [],
              'rho' : [],
              'h'   : [],
              'u'   : [],
              'frac': []}
    
    for i, comp in enumerate(system):
        # Update the m_dot based on pipe splits and merges
        if type(comp) == tuple:
            m_dot = m_dot * comp[1] / comp[-1]
        else:
            # Propagate the state through the specific component solver
            component_result = comp.solve_H2_state(states, T_amb, m_dot, PLOT=False, system=system, i=i)
            
            states['p'].append(component_result['p'])
            states['T'].append(component_result['T'])
            states['rho'].append(component_result['rho'])
            states['h'].append(component_result['h'])
            states['u'].append(component_result['u'])
            states['frac'].append(component_result['frac'])

    return states, m_dot

# =============================================================================
# Displays the computed states in a clean, hierarchical CLI tree format.
# =============================================================================
def print_tree(states):

    tree = Tree("\n[bold blue]System States")
    
    for key, values in states.items():
        branch = tree.add(f"[bold red]{key.upper()}")
        for comp_idx, data in enumerate(values):
            comp_node = branch.add(f"[bold yellow]Component {comp_idx}")
            comp_node.add(str(data))
            
    #rich_print(tree)

# =============================================================================
# Visualizes the pressure, temperature, density, and enthalpy profiles.
# Background gradient indicates the phase fraction (liquid to gas).
# =============================================================================
def plot_states(states):
  
    flat_states = {}
    for prop in ['p', 'T', 'rho', 'h', 'frac']:
        temp_list = []
        for component_data in states[prop]:
            for value in component_data:
                temp_list.append(value)
        flat_states[prop] = temp_list

    # Prepare phase-fraction background gradient
    frac_arr = np.array(flat_states['frac'])
    gradient = np.tile(frac_arr, (100, 1)) 

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    properties = ['p', 'T', 'rho', 'h']
    titles = ['Pressure (Pa)', 'Temperature (K)', 'Density (kg/m³)', 'Enthalpy (J/kg)']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']

    for i, prop in enumerate(properties):
        y_min   = min(flat_states[prop])
        y_max   = max(flat_states[prop])
        margin  = (y_max - y_min) * 0.05
        
        # Overlay phase map (Blue = Liquid, Red = Gas)
        axes[i].imshow(gradient, aspect='auto', cmap='RdYlBu_r',
                       vmin=0, vmax=1,
                       extent=[0, len(flat_states[prop]), y_min - margin, y_max + margin],
                       alpha=0.2)
        
        axes[i].plot(flat_states[prop], color=colors[i], marker=None, linestyle='-', linewidth=2)
        axes[i].set_title(titles[i])
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].set_ylabel(titles[i])
        axes[i].set_xlabel("Total System Step (Index)")

    fig.suptitle('Hydrogen State Profile (Gradient: Blue=Liquid, Red=Gas)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
       # Load the component cooling requirements from the propulsion json file
       path = str(root / "Propulsion/only_cooling_results.json")
       with open(path, 'r') as file:
              comps = json.load(file)

       for current_phase, current_mdot in zip(c.normal_phases, c.normal_m_dots):
              print("\n" + "*"*60)
              print(f" STARTING SIMULATION: {current_phase.upper()} ".center(60, "*"))
              print("*"*60)
              component_position = {}
              for key, value in comps[current_phase].items(): 
                     if not isinstance(value, dict):
                            continue
                     # Sort cooling locations by float
                     sorted_keys = sorted(value, key=float)
                     component_position[key] = sorted_keys
              # Define system topology as a sequential list of objects
              system = [
              Tank(),

              ('Split', 1, 2),

              Valve(name      =  'check'),
              
              Pipe(length     =  0.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  0.71),

              Valve(name      =  'shutoff'),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Valve(name      =  'shutoff'),

              Pipe(length     =  0.5),
              
              Pump(target_p   =  28*100000, 
              diameter   =  0.012),
              
              Pipe(length     =  12.62),

              Valve(name      =  'shutoff'),

              Corner(N_bend   =  1, 
                     curv     =  2.5),

              Valve(name      =  'shutoff'),
              
              Pipe(length     =  8.45),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  0.5), 
              
              COOL(name       = 'hts_gen', 
              location   = component_position['hts_gen'][0],
              phase = current_phase),

              Corner(N_bend   =  1, 
                     curv     =  2.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),
              
              Pipe(length     =  1.0), 
              
              COOL(name       = 'hts_pow', 
              location   = component_position['hts_pow'][0],
              phase = current_phase),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  0.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  5.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  0.5),

              COOL(name       = 'hts_pow', 
              location   = component_position['hts_pow'][1],
              phase = current_phase),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Corner(N_bend   =  1, 
                     curv     =  2.5),

              COOL(name       = 'dc_ac', 
              location   = component_position['dc_ac'][1],
              phase = current_phase),

              Pipe(length     =  0.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  5.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  0.5),

              COOL(name       = 'dc_ac', 
              location   = component_position['dc_ac'][0],
              phase = current_phase),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Corner(N_bend   =  1, 
                     curv     =  2.5),

              Pipe(length     =  1.0),

              COOL(name       = 'ac_dc', 
              location   = component_position['ac_dc'][0],
              phase = current_phase),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  1.0),

              COOL(name       = 'bus', 
              location   = component_position['bus'][0],
              phase = current_phase),

              Corner(N_bend   =  1, 
                     curv     =  2.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  1.0)
              ]
       
              # Execute simulation and display results
              states, final_mdot = solve_system(system, m_dot=current_mdot, T_amb=c.T_amb)
               # print_tree(states)
    
              # --- Extract and print the final state values ---
              final_T   = states['T'][-1][-1]
              final_p   = states['p'][-1][-1]
              final_rho = states['rho'][-1][-1]
              final_h   = states['h'][-1][-1]

              print("\n" + "="*50)
              print("FINAL FLUID STATE AT SYSTEM OUTLET".center(50))
              print("="*50)
              print(f"Temperature  :  {final_T:.2f} K")
              print(f"Pressure     :  {final_p:.2f} Pa  ({final_p/100000:.2f} bar)")
              print(f"Density      :  {final_rho:.2f} kg/m³")
              print(f"Enthalpy     :  {final_h:.2f} J/kg")
              print("="*50 + "\n")
              # -------------------------------------------------------

              # Save to JSON
              save_results_to_json(current_phase, final_T, final_p, final_rho, final_h, final_mdot)

              plot_states(states)