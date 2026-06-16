from pathlib import Path
import sys

# Set up paths to ensure we can import local modules
folder = Path(__file__).resolve().parent
sys.path.append(str(folder))

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

# Import your components from h2_components and configuration from system_config
from h2_components import Tank, Pipe, Pump, Corner, COOL, Valve, solve_system

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
def plot_states(states, phase_name, system):
  
    flat_states = {prop: [] for prop in ['p', 'T', 'rho', 'h', 'frac']}
    for prop in flat_states.keys():
        for component_data in states[prop]:
            flat_states[prop].extend(component_data)

    # --- MAP INDICES TO COMPONENTS ---
    state_idx = 0
    current_step = 0
    cool_spans = []

    for comp in system:
        if isinstance(comp, tuple): 
            continue # Skip pipe splits/merges

        comp_len = len(states['p'][state_idx])
        
        if comp.__class__.__name__ == 'COOL':
            cool_spans.append({
                'name': comp.name,
                'start': current_step,
                'end': current_step + comp_len,
                'mid': current_step + (comp_len / 2)
            })
        
        current_step += comp_len
        state_idx += 1
    # ---------------------------------

    # Prepare phase-fraction background gradient
    frac_arr = np.array(flat_states['frac'])
    gradient = np.tile(frac_arr, (100, 1)) 

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    
    properties = ['p', 'T', 'rho', 'h']
    titles = ['Pressure (Pa)', 'Temperature (K)', 'Density (kg/m³)', 'Enthalpy (J/kg)']
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd'] 

    for i, prop in enumerate(properties):
        y_min   = min(flat_states[prop])
        y_max   = max(flat_states[prop])
        
        margin  = (y_max - y_min) * 0.08 if (y_max - y_min) > 0 else 1.0

        # Overlay phase map (Blue = Liquid, Red = Gas)
        axes[i].imshow(gradient, aspect='auto', cmap='RdYlBu_r',
                       vmin=0, vmax=1,
                       extent=[0, len(flat_states[prop]), y_min - margin, y_max + margin],
                       alpha=0.15) 
        
        axes[i].plot(flat_states[prop], color=colors[i], marker=None, linestyle='-', linewidth=2.5)
        
        # --- DRAW COOL COMPONENT HIGHLIGHTS & NUMBERS ---
        for idx, span in enumerate(cool_spans):
            # Subtract 1 to target the exact inlet state BEFORE the thermodynamic jump
            inlet_idx = span['start'] - 1
            inlet_idx = max(0, inlet_idx) 
            
            # Draw the dashed line
            axes[i].axvline(x=inlet_idx, color='dimgray', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
            
            # Place a small sequence number directly above the line (y=1.02 puts it just above the top axis border)
            axes[i].text(inlet_idx, 1.02, str(idx + 1), transform=axes[i].get_xaxis_transform(),
                         ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444444', clip_on=False)
        # ---------------------------------------------------

        # Bumped pad to 20 to clear the new numbers
        axes[i].set_title(titles[i], pad=20, fontsize=14, fontweight='bold', color='#333333')
        
        axes[i].grid(True, linestyle=':', alpha=0.7, color='gray') 
        axes[i].set_ylabel(titles[i], fontsize=12)
        axes[i].set_ylim(y_min - margin, y_max + margin)
        axes[i].set_xlim(0, len(flat_states[prop]))
        axes[i].set_xlabel("Total System Step (Index)", fontsize=12, labelpad=8)
        axes[i].tick_params(axis='x', labelbottom=True)

    fig.tight_layout(pad=2.0, h_pad=4.0, w_pad=2.0)
    
    plt.show()

if __name__ == "__main__":
       # Load the component cooling requirements from the propulsion json file
       path = str(root / "Propulsion/only_cooling_results.json")
       with open(path, 'r') as file:
              comps = json.load(file)

       HEX_areas = None
       All_temps = {}

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
              
              Pipe(length     =  7.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  0.5), 
              
              COOL(name       = 'hts_gen', 
              location   = component_position['hts_gen'][0],
              phase = current_phase,
              areas = HEX_areas),

              Corner(N_bend   =  1, 
                     curv     =  2.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),
              
              Pipe(length     =  1.0), 
              
              COOL(name       = 'hts_pow', 
              location   = component_position['hts_pow'][0],
              phase = current_phase,
              areas = HEX_areas),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  0.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  7.5),

              COOL(name       = 'hts_pow', 
              location   = component_position['hts_pow'][1],
              phase = current_phase,
              areas = HEX_areas),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  0.5),

              COOL(name       = 'dc_ac', 
              location   = component_position['dc_ac'][1],
              phase = current_phase,
              areas = HEX_areas),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  7.5),

              COOL(name       = 'dc_ac', 
              location   = component_position['dc_ac'][0],
              phase = current_phase,
              areas = HEX_areas),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  1.0),

              COOL(name       = 'ac_dc', 
              location   = component_position['ac_dc'][0],
              phase = current_phase,
              areas = HEX_areas),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  1.0),

              COOL(name       = 'bus', 
              location   = component_position['bus'][0],
              phase = current_phase,
              areas = HEX_areas),

              Corner(N_bend   =  1, 
                     curv     =  2.5),

              Corner(N_bend   =  1,  
                     curv     =  2.5),

              Pipe(length     =  1.0)
              ]
       
              # Execute simulation and display results
              states, final_mdot, HEX_areas, Temps = solve_system(system, m_dot=current_mdot, T_amb=c.T_amb)
              All_temps[current_phase] = Temps
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

              plot_states(states, current_phase, system)

       filename_areas = "HEX_areas.json"
       with open(filename_areas, "w") as f:
              json.dump(HEX_areas, f, indent=4)

       filename_temps = "HEX_temps.json"
       with open(filename_temps, "w") as f:
              json.dump(All_temps, f, indent=4)