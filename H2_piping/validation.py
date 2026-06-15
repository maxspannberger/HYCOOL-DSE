from main import plot_states, print_tree
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
import pandas as pd
import CoolProp.CoolProp as CP
c = H2SystemConfig()  

def convert_to_si(data, is_setup_data=False):
    # Create a copy to avoid SettingWithCopy warnings
    si_df = data.copy()
    
    if is_setup_data: # mass_flow,heat_input,T_in,T_out,heat_input_per_area,length,d_in,d_out
        si_df['mass_flow'] = si_df['mass_flow'] * 0.453592                      # lb/s      --> kg/s
        si_df['heat_input'] = si_df['heat_input'] * 1055.06                     # BTU/s     --> Watts
        si_df['T_in'] = si_df['T_in'] * 5/9                                     # Rankine   --> Kelvin
        si_df['T_out'] = si_df['T_out'] * 5/9                                   # Rankine   --> Kelvin
        si_df['heat_input_per_area'] = si_df['heat_input_per_area'] * 1636183.5 # BTU/s/in² --> W/m²
        si_df['length'] = si_df['length'] * 0.0254                              # inches    --> m
        si_df['d_in'] = si_df['d_in'] * 0.0254                                  # inches    --> m
        si_df['d_out'] = si_df['d_out'] * 0.0254                                # inches    --> m
            
    else:
        si_df['l'] = si_df['l'] * 0.0245      # inch    --> meters
        si_df['p'] = si_df['p'] * 6894.76     # psi     --> pascals
        si_df['T'] = si_df['T'] * 5/9         # rankine --> kelvin
        si_df['rho'] = si_df['rho'] *16.0185  # lb/ft^3 --> kg/m^3     
        si_df['u'] = si_df['u'] * 0.3048      # ft/sec  --> m/sec
        
        
    # CONVERSIONS = {
    #     'length_in_m': 0.0254,
    #     'psi_to_pa': 6894.76,
    #     'rankine_to_k': 5/9,
    #     'lbft3_to_kgm3': 16.0185,
    #     'ftsec_to_msec': 0.3048
    # }
        
    return si_df

def plot_validation(states, phase_name, validation_df=None):
    # 1. Flatten the simulation data
    flat_states = {}
    for prop in ['p', 'T', 'rho', 'h', 'frac']:
        temp_list = []
        for component_data in states[prop]:
            for value in component_data:
                temp_list.append(value)
        flat_states[prop] = temp_list

    # 2. Setup Plotting
    frac_arr = np.array(flat_states['frac'])
    gradient = np.tile(frac_arr, (100, 1)) 
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    properties = ['p', 'T', 'rho', 'h']
    titles = ['Pressure (Pa)', 'Temperature (K)', 'Density (kg/m³)', 'Enthalpy (J/kg)']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
    
    # Map your DataFrame columns to the simulation properties
    val_map = {'p': 'p', 'T': 'T', 'rho': 'rho', 'h': None}

    # 3. Plotting Loop
    for i, prop in enumerate(properties):
        y_data = flat_states[prop]
        x_steps = np.arange(len(y_data))
        
        y_min, y_max = min(y_data), max(y_data)
        margin = (y_max - y_min) * 0.1
        
        # Overlay Phase Map
        axes[i].imshow(gradient, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1,
                       extent=[0, len(y_data), y_min - margin, y_max + margin], alpha=0.2)
        
        # Plot Simulation
        axes[i].plot(x_steps, y_data, color=colors[i], label='Simulation', linewidth=2)
        
        # 4. Overlay Validation Data with Interpolation
        if validation_df is not None and val_map[prop] in validation_df.columns:
            # Create mapping: scale validation length to simulation index range
            val_x = np.linspace(0, len(y_data) - 1, len(validation_df))
            axes[i].scatter(val_x, validation_df[val_map[prop]], 
                            color='black', marker='x', label='Validation', s=40, zorder=5)
            axes[i].legend()

        axes[i].set_title(titles[i])
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].set_ylabel(titles[i])
        axes[i].set_xlabel("System Step Index")

    fig.suptitle(f'Hydrogen State Profile ({phase_name.upper()})', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
       # Load the component cooling requirements from the propulsion json file

    HEX_areas = None
    All_temps = {}
       

    # Extract data from test set csv
    setup_data    = pd.read_csv('validation_set_run_2.csv', skiprows=0, nrows=1)
    state_data    = pd.read_csv('validation_set_run_2.csv', skiprows=3)
    
    # Remove NaN from state_data
    state_data    = state_data.dropna(axis=1, how='all')
    
    # Convert to si units
    setup_data_si = convert_to_si(setup_data, is_setup_data=True)
    state_data_si = convert_to_si(state_data)
    
    print(setup_data_si)
    print(state_data)

    # Calculate segment lenth between first and last data point
    length = state_data_si['l'].iloc[-1] - state_data_si['l'].iloc[0]
    
    Q_set = setup_data_si['heat_input_per_area'].item() * setup_data_si['d_in'].item() * np.pi * length
    q_set = Q_set / setup_data_si['mass_flow'].item()
    
#   Define the system 
# =============================================================================       
    system = [
    Pipe(length=length, diameter=setup_data_si['d_in'].iloc[0], q_set=q_set)
    ]
# ============================================================================= 
    input_states = {'p'   : [np.array([state_data_si['p'].iloc[0]])],
                    'T'   : [np.array([state_data_si['T'].iloc[0]])],
                    'rho' : [np.array([state_data_si['rho'].iloc[0]])],
                    'h'   : [np.array([CP.PropsSI('H', 'P', state_data_si['p'].iloc[0], 'T', state_data_si['T'].iloc[0], c.fluid)])],
                    'u'   : [np.array([state_data_si['u'].iloc[0]])],
                    'frac': [np.array([0])]}      

    states, final_mdot, HEX_areas, Temps = solve_system(system, m_dot=setup_data_si['mass_flow'].iloc[0], T_amb=c.T_amb, input_states=input_states)
    
    plot_validation(states, 'Validation', validation_df=state_data_si)
