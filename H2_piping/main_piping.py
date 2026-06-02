from pathlib import Path
import sys
root = Path(__file__).resolve().parent
sys.path.append(str(root))

from h2_components import Tank, Pipe
from rich import print as rich_print
from rich.tree import Tree

'''
The wall of the components Pipe and Tank should be defined in the following manner.
 
- Wall contains a list of tupples. 
- Within the tupple the material is specified at index 0 and the thickness
  at index 1. 
- The materials should be oredered from inner tube to outer tube

For example: wall = [('ss-316l', 0.01), ('polyurethene', 0.02)]
eg. the inner layer is ss-326l and the outer layer is polyurethene

'''

def solve_system(system, m_dot, T_amb):
    states = {'p'   : [],
              'T'   : [],
              'rho' : [],
              'h'   : []}
    
    
    for comp in system:
        component_result = comp.solve_H2_state(states, T_amb, m_dot, PLOT=False)
        
        states['p'].append(component_result['p'])
        states['T'].append(component_result['T'])
        states['rho'].append(component_result['rho'])
        states['h'].append(component_result['h'])
        
    return states

def print_tree(states):
    tree = Tree("[bold blue]System States")
    
    for key, values in states.items():
        branch = tree.add(f"[bold magenta]{key.upper()}")
        for comp_idx, data in enumerate(values):
            comp_node = branch.add(f"Component {comp_idx}")
            # Format the array for readability
            comp_node.add(str(data))
            
    rich_print(tree)


if __name__ == "__main__":
    wall = [('ss-316l',      0.01), 
            ('polyurethene', 0.02)]

    system = [
        Tank(diameter=0.1, wall=wall), 
        Pipe(position=1, length=1.0, diameter=0.1, wall=wall, segments=10,
             N=3, N_bar=0.1, P_mli=0.1)
        ]
    
    states = solve_system(system, 0.03, 307)
    print_tree(states)



          

