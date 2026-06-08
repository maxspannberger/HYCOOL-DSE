from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.path.append(str(folder))

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

# Import your components from h2_components and configuration from system_config
from h2_components import Tank, Pipe, Pump, Corner, COOL

from rich import print as rich_print
from rich.tree import Tree
import matplotlib.pyplot as plt
import numpy as np
import json
from system_config import H2SystemConfig as c

'''
===============================================================================
The wall of the components Pipe and Tank should be defined in the following manner.
 
- Wall contains a list of tupples. 
- Within the tupple the material is specified at index 0 and the thickness
  at index 1. 
- The materials should be oredered from inner tube to outer tube

For example: wall = [('ss-316l', 0.01), ('polyurethene', 0.02)]
eg. the inner layer is ss-326l and the outer layer is polyurethene
===============================================================================
'''

def solve_system(system, m_dot, T_amb):
    states = {'p'   : [],
              'T'   : [],
              'rho' : [],
              'h'   : [],
              'frac': []}
    
    for i, comp in enumerate(system):
        # Update the m_dot based on pipe splits and merges
        if type(comp) == tuple:
            m_dot = m_dot * comp[1] / comp[-1]
        else:
            component_result = comp.solve_H2_state(states, T_amb, m_dot, PLOT=False, system=system, i=i)
            
            states['p'].append(component_result['p'])
            states['T'].append(component_result['T'])
            states['rho'].append(component_result['rho'])
            states['h'].append(component_result['h'])
            states['frac'].append(component_result['frac'])
    print(m_dot)   
    return states


def print_tree(states):
    tree = Tree("\n[bold blue]System States")
    
    for key, values in states.items():
        branch = tree.add(f"[bold red]{key.upper()}")
        for comp_idx, data in enumerate(values):
            comp_node = branch.add(f"[bold yellow]Component {comp_idx}")
            comp_node.add(str(data))
            
    rich_print(tree)

def plot_states(states):
    flat_states = {}
    for prop in ['p', 'T', 'rho', 'h', 'frac']:
        temp_list = []
        for component_data in states[prop]:
            for value in component_data:
                temp_list.append(value)
        flat_states[prop] = temp_list

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

# Load the component cooling requirements from the propulsion json file
path = str(root / "Propulsion/component_sizing_results.json")
with open(path, 'r') as file:
    comps = json.load(file)

component_order = {}
for key, value in comps.items(): 
    if key == "total":
        continue
    sorted_keys = sorted(value, key=int)
    component_order[key] = sorted_keys

if __name__ == "__main__":
    wall = [('ss-316l',      0.01), 
            ('polyurethene', 0.02)]


    # Distribute the configuration tracking instance down into each custom layout element
    system = [
        Tank(), 
        
        Pipe(length     =  0.5),
         
        Pump(target_p   =  5*100000, 
             diameter   =  0.02),
        
        Pipe(length     =  12),
        
        ('Split', 1, 2),
        
        Pipe(length     =  12.0), 
        
        Pipe(length     =  4.0), 
        
        COOL(name       = 'hts_gen', 
             location   = component_order['hts_gen'][0]),
        
        Pipe(length     =  2.0), 

        COOL(name       = 'bus', 
             location   = component_order['bus'][0]),
        
        Pipe(length     =  2.0), 
        
        COOL(name       = 'ac_dc', 
             location   = component_order['ac_dc'][0]),
        
        Pipe(length     = 4.0), 
        
        COOL(name       = 'dc_ac', 
             location   = component_order['dc_ac'][0]),

        COOL(name       = 'ac_dc', 
             location   = component_order['ac_dc'][0]),
        
        Pipe(length     =  2.0),  
        
        COOL(name       = 'hts_pow', 
             location   = component_order['hts_pow'][0]),
        
        Corner(N_bend   =  10, 
               diameter =  0.02, 
               curv     =  2.5)
        ]
    
    states = solve_system(system, m_dot=c.m_dot, T_amb=c.T_amb)
    print_tree(states)
    plot_states(states)