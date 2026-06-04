from pathlib import Path
import sys
root = Path(__file__).resolve().parent
sys.path.append(str(root))

from h2_components import Tank, Pipe, Corner
from rich import print as rich_print
from rich.tree import Tree
import matplotlib.pyplot as plt
import numpy as np

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
        component_result = comp.solve_H2_state(states, T_amb, m_dot, PLOT=False)
        
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
            # Format the array for readability
            comp_node.add(str(data))
            
    rich_print(tree)

def plot_states(states):
    # Flatten the lists into simple lists for plotting
    flat_states = {}
    for prop in ['p', 'T', 'rho', 'h', 'frac']:
        temp_list = []
        for component_data in states[prop]:
            for value in component_data:
                temp_list.append(value)
        flat_states[prop] = temp_list

    # Prepare the gradient
    # No clipping needed here if we use vmin/vmax in imshow
    frac_arr = np.array(flat_states['frac'])
    gradient = np.tile(frac_arr, (100, 1)) 

    # Plot the flattened data
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    properties = ['p', 'T', 'rho', 'h']
    titles = ['Pressure (Pa)', 'Temperature (K)', 'Density (kg/m³)', 'Enthalpy (J/kg)']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']

    for i, prop in enumerate(properties):
        # Set vmin=0 and vmax=1 to lock the colors to the full range
        axes[i].imshow(gradient, aspect='auto', cmap='RdYlBu_r', 
                       vmin=0, vmax=1,
                       extent=[0, len(flat_states[prop]), min(flat_states[prop]), max(flat_states[prop])],
                       alpha=0.2)
        
        # Plot the main data line
        axes[i].plot(flat_states[prop], color=colors[i], marker=None, linestyle='-', linewidth=2)
        axes[i].set_title(titles[i])
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].set_ylabel(titles[i])
        axes[i].set_xlabel("Total System Step (Index)")

    fig.suptitle('Hydrogen State Profile (Gradient: Blue=Liquid, Red=Gas)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    wall = [('ss-316l',      0.01), 
            ('polyurethene', 0.02)]

    system = [
        Tank(diameter=0.1, wall=wall), 
        
        Pipe(position=1, length=64.0, diameter=0.02, wall=wall, segments=200,
             N=11, N_bar=5.5, P_mli=0.001, curv=2.5),
        
        Corner(position=1, N_bend=10, diameter=0.02, curv=2.5)
        ]
    
    states = solve_system(system, 0.03, 313)
    print_tree(states)
    plot_states(states)



          

