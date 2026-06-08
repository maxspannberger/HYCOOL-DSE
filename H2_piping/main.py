from pathlib import Path
import sys
folder = Path(__file__).resolve().parent
sys.path.append(str(folder))

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

# Import your components from h2_components and configuration from system_config
from h2_components import Tank, Pipe, Pump, Corner, COOL
from system_config import H2SystemConfig

from rich import print as rich_print
from rich.tree import Tree
import matplotlib.pyplot as plt
import numpy as np
import json

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
    
    for comp in system:
        # Update the m_dot based on pipe splits and merges
        if type(comp) == tuple:
            m_dot = m_dot * comp[1] / comp[-1]
        else:
            component_result = comp.solve_H2_state(states, T_amb, m_dot, PLOT=False, system=system)
            
            states['p'].append(component_result['p'])
            states['T'].append(component_result['T'])
            states['rho'].append(component_result['rho'])
            states['h'].append(component_result['h'])
            states['frac'].append(component_result['frac'])
        
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
        axes[i].imshow(gradient, aspect='auto', cmap='RdYlBu_r', 
                       vmin=0, vmax=1,
                       extent=[0, len(flat_states[prop]), min(flat_states[prop]), max(flat_states[prop])],
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
    sorted_keys = sorted(value)
    component_order[key] = sorted_keys

if __name__ == "__main__":
    wall = [('ss-316l',      0.01), 
            ('polyurethene', 0.02)]

    # Instantiate custom baseline parameters for the configuration tracking class
    custom_config = H2SystemConfig(
        fluid              = 'Hydrogen',
        divergence_penalty = 1e9,
        pipe_mli_eps       = 0.03,
        pipe_default_d     = 0.02,
        pipe_default_N     = 10,
        pipe_default_N_bar = 5.5
    )

    # Distribute the configuration tracking instance down into each custom layout element
    system = [
        Tank(diameter   =  0.1, 
             p          =  1.0*101325, 
             T          =  15), 
        
        Pipe(length     =  0.5),
         
        Pump(target_p   =  20*100000, 
             diameter   =  0.02, 
             efficiency =  0.65),
        
        Pipe(length     =  0.5),
        
        ('Split', 1, 2),
        
        Pipe(length     =  64.0, 
             segments   =  200,
             eps_pipe   =  1.5), 
        
        ('Split', 2, 4),
        
        Pipe(length     =  64.0, 
             segments   =  200,
             eps_pipe   =  1.5), 
        
        COOL(name       = 'hts_gen', 
             location   = component_order['hts_gen'][0]),
        
        Pipe(length     =  64.0, 
             segments   =  200,
             eps_pipe   =  1.5), 

        COOL(name       = 'bus', 
             location   = component_order['bus'][0]),
        
        Pipe(length     =  64.0, 
             segments   =  200,
             eps_pipe   =  1.5), 
        
        COOL(name       = 'ac_dc', 
             location   = component_order['ac_dc'][0]),
        
        Pipe(length     =  64.0, 
             segments   =  200,
             eps_pipe   =  1.5), 
        
        COOL(name       = 'dc_ac', 
             location   = component_order['dc_ac'][0]),

        COOL(name       = 'ac_dc', 
             location   = component_order['ac_dc'][0]),
        
        Pipe(length     =  64.0, 
             segments   =  200,
             eps_pipe   =  1.5),                           
        
        ('Converge', 2, 1),
        
        COOL(name       = 'hts_gen', 
             location   = component_order['hts_gen'][0]),
        
        Corner(N_bend   =  10, 
               diameter =  0.02, 
               curv     =  2.5)
        ]
    
    states = solve_system(system, m_dot=0.06, T_amb = 317)
    print_tree(states)
    plot_states(states)