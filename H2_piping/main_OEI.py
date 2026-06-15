from pathlib import Path
import sys
import numpy as np
import json
import CoolProp.CoolProp as CP
from rich import print as rich_print
from rich.tree import Tree
import matplotlib.pyplot as plt

# Set up paths to ensure we can import local modules
folder = Path(__file__).resolve().parent
sys.path.append(str(folder))
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from H2_piping.h2_components import Tank, Pipe, Pump, Corner, COOL, Valve, calc_frac
from H2_piping.system_config import H2SystemConfig
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
# Visualization
# =============================================================================
def solve_OEI_system(system, m_dot, T_amb, branch_name="", show=False):
    # Add 'u': [] to the initialization
    states = {'p': [], 'T': [], 'rho': [], 'h': [], 'frac': [], 'u': []}
    
    Temps = {}

    for i, comp in enumerate(system):
        if type(comp) == tuple:
            m_dot = m_dot * comp[1] / comp[-1]
        else:
            if hasattr(comp, 'name') and comp.name in ['hts_gen', 'hts_pow']:
                if show:
                    print(f"[{branch_name}] Mass flow entering {comp.name.upper()} ({comp.location}): {m_dot:.5f} kg/s")

            component_result = comp.solve_H2_state(states, T_amb, m_dot, PLOT=False, system=system, i=i)
            states['p'].append(component_result['p'])
            states['T'].append(component_result['T'])
            states['rho'].append(component_result['rho'])
            states['h'].append(component_result['h'])
            states['frac'].append(component_result['frac'])
            states['u'].append(component_result['u']) 

            if "temperature" in component_result:
                 if comp.name not in Temps:
                     Temps[comp.name] = {}
                 Temps[comp.name][comp.location] = component_result['temperature']

    if show:    
        print(f"[{branch_name}] Ending Branch mass flow: {m_dot:.5f} kg/s")

    return states, m_dot, None, Temps


def plot_combined_states(states_W, states_F, phase_name):
    flat_W = {}
    flat_F = {}
    for prop in ['p', 'T', 'rho', 'h', 'frac']:
        flat_W[prop] = [value for component_data in states_W[prop] for value in component_data]
        flat_F[prop] = [value for component_data in states_F[prop] for value in component_data]

    frac_W_arr = np.array(flat_W['frac'])
    frac_F_arr = np.array(flat_F['frac'])
    gradient = np.vstack((np.tile(frac_W_arr, (50, 1)), np.tile(frac_F_arr, (50, 1)))) 

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    properties = ['p', 'T', 'rho', 'h']
    titles = ['Pressure (Pa)', 'Temperature (K)', 'Density (kg/m³)', 'Enthalpy (J/kg)']

    for i, prop in enumerate(properties):
        y_min   = min(min(flat_W[prop]), min(flat_F[prop]))
        y_max   = max(max(flat_W[prop]), max(flat_F[prop]))
        margin  = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 1.0
        y_mid   = (y_max + y_min) / 2.0
        
        axes[i].imshow(gradient, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1,
                       extent=[0, len(flat_W[prop]), y_min - margin, y_max + margin], alpha=0.25)
        
        axes[i].axhline(y_mid, color='black', linestyle='-.', linewidth=1.2, alpha=0.4)
        axes[i].text(len(flat_W[prop])*0.01, y_max, " Working Wing Phase Map", fontsize=8, color='black', alpha=0.7, va='top')
        axes[i].text(len(flat_W[prop])*0.01, y_min, " Failed Wing Phase Map", fontsize=8, color='black', alpha=0.7, va='bottom')

        axes[i].plot(flat_W[prop], color='tab:blue', linestyle='-', linewidth=2.5, label='Working Wing')
        axes[i].plot(flat_F[prop], color='tab:red', linestyle='--', linewidth=2.0, label='Failed Wing')
        
        axes[i].set_title(titles[i])
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].set_ylabel(titles[i])
        axes[i].set_xlabel("Total System Step (Index)")
        axes[i].legend()

    fig.text(0.5, 0.01, "Background Gradient: Top Half = Working Wing Phase | Bottom Half = Failed Wing Phase (Blue=Liquid, Red=Gas)", ha='center', fontsize=10, style='italic')
    fig.suptitle(f'Hydrogen State Profile Comparison (Phase: {phase_name.upper()})', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main_H2_OEI(comps=None, sizes=None, All_temps=None, HEX_areas=None, show=False, write=False, oei_phases=None, oei_m_dots=None):
    if comps is None:
        path = str(root / "Propulsion/only_cooling_results.json")
        with open(path, 'r') as file:
            comps = json.load(file)
    if sizes is None:
            path = root / "Propulsion" / "only_sizing_results.json"
            with open(path, 'r') as file:
                    sizes = json.load(file)
    if All_temps is None:
        filename = "HEX_temps.json"
        with open(filename, 'r') as f:
            All_temps = json.load(f)
    if HEX_areas is None:
        filename = "HEX_areas.json"
        with open(filename, 'r') as f:
            HEX_areas = json.load(f)
    if oei_phases is None:
        oei_phases = c.oei_phases
    if oei_m_dots is None:
        oei_m_dots = c.oei_m_dots

    for current_phase, current_mdot in zip(oei_phases, oei_m_dots):
        if show:
            print("\n" + "*"*60)
            print(f" STARTING SIMULATION: {current_phase.upper()} ".center(60, "*"))
            print("*"*60)

        component_position = {}
        data_per_condition = {}
        for key, value in comps[current_phase].items(): 
            if key == "total": continue
            component_position[key] = sorted(value, key=float)

        # 1. Mass Flow Split Configuration
        if current_phase == 'OEI_gt':
            cool_data = comps[current_phase]
            Q_working = sum(cool_data.get('hts_pow', {}).values()) + sum(cool_data.get('dc_ac', {}).values()) + \
                        sum(cool_data.get('hts_gen', {}).values()) + sum(cool_data.get('ac_dc', {}).values()) + \
                        sum(cool_data.get('bus', {}).values()) 
            Q_failed  = sum(cool_data.get('hts_pow', {}).values()) + sum(cool_data.get('dc_ac', {}).values()) + \
                        sum(cool_data.get('bus', {}).values())
            frac_W = Q_working / (Q_working + Q_failed)
            frac_F = Q_failed / (Q_working + Q_failed)
        else:
            frac_W = 0.50
            frac_F = 0.50

        # 2. Build Branches (Instantiated with current phase)
        def create_oei_branch(frac):
            return [
                Tank(), ('Split', frac, 1.0), Valve(name='check'), Pipe(length=0.5), Corner(N_bend=1, curv=2.5),
                Pipe(length=0.71), Valve(name='shutoff'), Corner(N_bend=1, curv=2.5), Valve(name='shutoff'),
                Pipe(length=0.5), Pump(target_p=28*100000, diameter=0.012), Pipe(length=12.62), Valve(name='shutoff'),
                Corner(N_bend=1, curv=2.5), Valve(name='shutoff'), Pipe(length=8.45), Corner(N_bend=1, curv=2.5), Pipe(length=0.5), 
                COOL(name='hts_gen', location=component_position['hts_gen'][0], phase=current_phase, areas=HEX_areas, comps=comps, sizes=sizes),
                Corner(N_bend=1, curv=2.5), Corner(N_bend=1, curv=2.5), Pipe(length=1.0), 
                COOL(name='hts_pow', location=component_position['hts_pow'][0], phase=current_phase, areas=HEX_areas, comps=comps, sizes=sizes),
                Corner(N_bend=1, curv=2.5), Corner(N_bend=1, curv=2.5), Pipe(length=0.5), Corner(N_bend=1, curv=2.5), Pipe(length=5.5), Corner(N_bend=1, curv=2.5), Pipe(length=0.5),
                COOL(name='hts_pow', location=component_position['hts_pow'][1], phase=current_phase, areas=HEX_areas, comps=comps, sizes=sizes),
                Corner(N_bend=1, curv=2.5), Corner(N_bend=1, curv=2.5),
                COOL(name='dc_ac', location=component_position['dc_ac'][1], phase=current_phase, areas=HEX_areas, comps=comps, sizes=sizes),
                Pipe(length=0.5), Corner(N_bend=1, curv=2.5), Pipe(length=5.5), Corner(N_bend=1, curv=2.5), Pipe(length=0.5),
                COOL(name='dc_ac', location=component_position['dc_ac'][0], phase=current_phase, areas=HEX_areas, comps=comps, sizes=sizes),
                Corner(N_bend=1, curv=2.5), Corner(N_bend=1, curv=2.5), Pipe(length=1.0),
                COOL(name='ac_dc', location=component_position['ac_dc'][0], phase=current_phase, areas=HEX_areas, comps=comps, sizes=sizes),
                Corner(N_bend=1, curv=2.5), Pipe(length=1.0),
                COOL(name='bus', location=component_position['bus'][0], phase=current_phase, areas=HEX_areas, comps=comps, sizes=sizes),
                Corner(N_bend=1, curv=2.5), Corner(N_bend=1, curv=2.5), Pipe(length=1.0)
            ]

        system_W = create_oei_branch(frac_W)
        system_F = create_oei_branch(frac_F)

        # 3. Dynamic Kill Switches
        for comp_W, comp_F in zip(system_W, system_F):
            if isinstance(comp_W, COOL):
                if current_phase == 'OEI_gt':
                    if comp_F.name in ['hts_gen', 'ac_dc']: comp_F.Q_dot = 0.0
                elif current_phase == 'OEI_mot':
                    if comp_F.name in ['hts_pow', 'dc_ac'] and comp_F.location == component_position[comp_F.name][0]: comp_F.Q_dot = 0.0
                elif current_phase == 'OEI_bus':
                    if comp_F.name == 'bus': comp_F.Q_dot = 0.0

        # 4. Execute Simulation
        states_W, final_mdot_W, _, Temps_W = solve_OEI_system(system_W, m_dot=current_mdot, T_amb=c.T_amb, branch_name="Working Wing", show=show)
        states_F, final_mdot_F, _, Temps_F = solve_OEI_system(system_F, m_dot=current_mdot, T_amb=c.T_amb, branch_name="Failed Wing", show=show)
        
        All_temps[current_phase] = {}
        All_temps[current_phase]["W"] = Temps_W
        All_temps[current_phase]["F"] = Temps_F

        # 5. Output and JSON Export
        T_W, p_W, rho_W, h_W = states_W['T'][-1][-1], states_W['p'][-1][-1], states_W['rho'][-1][-1], states_W['h'][-1][-1]
        T_F, p_F, rho_F, h_F = states_F['T'][-1][-1], states_F['p'][-1][-1], states_F['rho'][-1][-1], states_F['h'][-1][-1]

        if current_phase == 'OEI_gt':
            m_dot_mix = final_mdot_W + final_mdot_F
            h_mix = (final_mdot_W * h_W + final_mdot_F * h_F) / m_dot_mix
            p_mix = min(p_W, p_F) 
            T_mix = CP.PropsSI('T', 'P', p_mix, 'H', h_mix, c.fluid)
            rho_mix = CP.PropsSI('D', 'P', p_mix, 'H', h_mix, c.fluid)
            
            if show:
                print(f"\nFinal Engine Inlet Temp (Merged): {T_mix:.2f} K")

            data_per_condition[current_phase] = {
                "Temperature_K": round(float(T_mix), 2),
                "Pressure_Pa": round(float(p_mix), 2),
                "Density_kg_m3": round(float(rho_mix), 2),
                "Enthalpy_J_kg": round(float(h_mix), 2),
                "Final_MassFlow_kg_s": round(float(m_dot_mix), 5)
            }
        else:
            if show:
                print(f"\nWorking Wing Inlet Temp: {T_W:.2f} K")
                print(f"Failed Wing Inlet Temp: {T_F:.2f} K")

            data_per_condition[f"{current_phase}_Working"] = {
                "Temperature_K": round(float(T_mix), 2),
                "Pressure_Pa": round(float(p_mix), 2),
                "Density_kg_m3": round(float(rho_mix), 2),
                "Enthalpy_J_kg": round(float(h_mix), 2),
                "Final_MassFlow_kg_s": round(float(m_dot_mix), 5)
            }
            data_per_condition[f"{current_phase}_Failed"] = {
                "Temperature_K": round(float(T_mix), 2),
                "Pressure_Pa": round(float(p_mix), 2),
                "Density_kg_m3": round(float(rho_mix), 2),
                "Enthalpy_J_kg": round(float(h_mix), 2),
                "Final_MassFlow_kg_s": round(float(m_dot_mix), 5)
            }
        if write:
            results_file = root / "Propulsion" / "final_states.json"
            with open(results_file, 'w') as f:
                json.dump(data_per_condition, f, indent=4)

        if show:
            plot_combined_states(states_W, states_F, current_phase)

    if write:
        filename_temps = "HEX_temps.json"
        with open(filename_temps, "w") as f:
                json.dump(All_temps, f, indent=4)

    all_H2_results = {
        "final_states": data_per_condition,
        "areas": HEX_areas,
        "temperatures": All_temps
    }

    return all_H2_results

# =============================================================================
# MAIN RUNTIME ENVIRONMENT
# =============================================================================
if __name__ == "__main__":
    all_H2_results = main_H2_OEI(show=False, write=False)
    print(all_H2_results)