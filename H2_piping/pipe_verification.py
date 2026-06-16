from pathlib import Path
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

# Set up paths to ensure we can import local modules
folder = Path(__file__).resolve().parent
sys.path.append(str(folder))

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

# Import your native components and configuration
from h2_components import Tank, Pipe, Corner, COOL, Pump, solve_system
from system_config import H2SystemConfig
c = H2SystemConfig()  

# =============================================================================
# Visualizes the pressure, temperature, density, and enthalpy profiles.
# Background gradient indicates the phase fraction (liquid to gas).
# =============================================================================
def plot_states(states, phase_name):
  
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
        margin  = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 1.0
        
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

    fig.suptitle(f'Hydrogen State Profile (Test: {phase_name} | Gradient: Blue=Liquid, Red=Gas)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# MAIN RUNTIME ENVIRONMENT
# =============================================================================
if __name__ == "__main__":
    
    # Load the component cooling requirements to safely initialize the COOL class
    path = str(root / "Propulsion/only_cooling_results.json")
    with open(path, 'r') as file:
        comps = json.load(file)

    # Grab the location data for the component
    ref_phase = 'TO'
    component_position = {}
    for key, value in comps[ref_phase].items(): 
        if not isinstance(value, dict): continue
        component_position[key] = sorted(value, key=float)

    # Instantiate the COOL component normally using JSON data
    test_cool = COOL(
        name='hts_gen', 
        location=component_position['hts_gen'][0], 
        phase=ref_phase, 
        areas=None
    )

    # =========================================================================
    # MANUAL OVERRIDE ZONE
    # =========================================================================
    # 1. Overwrite the Mass Flow (kg/s)
    current_mdot = 0.0894/2   

    # 2. Overwrite the Heat Load directly into the COOL component 
    test_cool.Q_dot = 153.0*25 # [kW]  
    # =========================================================================

    print(f"Running System -> Mass Flow: {current_mdot} kg/s | Heat Load: {test_cool.Q_dot} kW")

    # Define the simplified test system topology
    system = [
        Tank(),
        Pump(target_p   =  28*100000, 
              diameter   =  0.012),
        Pipe(length=20.0, diameter=0.012),
        Corner(curv=2.5, diameter=0.012),
        test_cool,
        Pipe(length=10.0, diameter=0.012),
        test_cool,
        Pipe(length=10.0, diameter=0.012)
    ]
    
    # Execute simulation
    states, final_mdot, HEX_areas, Temps = solve_system(system, m_dot=current_mdot, T_amb=c.T_amb)
    
    # --- Extract and print the final state values ---
    final_T   = states['T'][-1][-1]
    final_p   = states['p'][-1][-1]
    final_rho = states['rho'][-1][-1]
    final_h   = states['h'][-1][-1]
    dp_total  = c.tank_p - final_p
    cool_L    = HEX_areas['hts_gen'][component_position['hts_gen'][0]]['pipe_length']

    print("\n" + "="*50)
    print("FINAL FLUID STATE AT SYSTEM OUTLET".center(50))
    print("="*50)
    print(f"Temperature      :  {final_T:.2f} K")
    print(f"Pressure         :  {final_p:.2f} Pa  ({final_p/100000:.2f} bar)")
    print(f"Total Press Drop :  {dp_total:.2f} Pa")
    print(f"Req. HEX Length  :  {cool_L:.2f} meters")
    print(f"Density          :  {final_rho:.2f} kg/m³")
    print(f"Enthalpy         :  {final_h:.2f} J/kg")
    print("="*50 + "\n")
    
    # Render the plots
    plot_states(states, "Manual Override Run")