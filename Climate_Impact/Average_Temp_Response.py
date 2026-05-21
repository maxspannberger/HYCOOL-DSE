import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

local_dir = Path(__file__).resolve().parent
if str(local_dir) not in sys.path:
    sys.path.insert(0, str(local_dir))

import pandas as pd
import ast

from Propulsion.efficiency import (
    FC_BAT_efficiency,
    GT_BAT_efficiency,
    GT_FC_efficiency,
    GT_GT_efficiency,
)
from General.component_parameters import component_params

import Outgoing_Longwave_Radiation as olr
import Potential_Vorticity as pv
from math import sin, cos, radians



# constants defined from literature and datasets
h = 7620 # m, based on the operating altitude of Dash 8 Q400 [Janes]
g0 = 9.80665 # m/s^2, standard gravity
latitude = 51.0 # degrees, based on the location of interest (central Europe/Germany)
f_day = 0.5 # day fraction, assuming 12 hours of daylight on average
f_ISSR  = 0.15 # ISSR fraction [Lamquin 2012]
pv = pv.pv_376 # PV at 376 hPa in PVU
olr = olr.olr # OLR in W/m2 
t = 238.62 # K, temperature altitude [standard atmosphere at 7620 m]
n = 80 # dayof the year, Spring Equinox (maybe replace with average over the year)
d_mission = 1000 # km, mission distance
ei_nox = 3*10**(-3) # kg/kg, 3 = lean premixed [Grewe 2016]; 4.5 = NOx emission index 2015 [Ponater 2006] 1.5 a 2050 predicted
r = 6356766 # m, Earth's radius
geopotential = h * g0 * (r/(r+h))**2 # m^2/s^2, geopotential at 7620 m
s = 1360 # W/m2, solar constant

# system-specific parameters to be changed based on the design of the aircraft and mission profile
#eta_prop = 0.6 # propulsion efficiency
m_h2 = 569 # kg, mass of hydrogen fuel
#E_mission = 500000 # kJ, energy required for the mission
turbine = True
fc_liquid_venting = False
#gravimetric energy density adjustment factor for baseline  system
f_energy_density = 120/43 # from individual densities in MJ/kg, 120 for LH2, 43 for Jet fuel

# =============================================================================
# Loading results from Class 2 and calculating the mission phase power and 
# energy requirements.
# =============================================================================

# Load the data
df = pd.read_csv(root / "outputs/class_ii_results.csv")
# Clean up whitespace (CSV exports often have hidden spaces in strings)
df['Section'] = df['Section'].str.strip()
df['Parameter'] = df['Parameter'].str.strip()

# Create a helper function to pull values safely
def get_param(parameter_name, prefer_section='Mission Power & Fuel'):
    # Prefer the mission section if the parameter appears multiple times
    mask_pref = (df['Parameter'] == parameter_name) & (df['Section'] == prefer_section)
    try:
        if mask_pref.any():
            val = df.loc[mask_pref, 'Value'].values[0]
        else:
            vals = df.loc[df['Parameter'] == parameter_name, 'Value'].values
            if len(vals) == 0:
                print(f"Error: Parameter '{parameter_name}' not found in CSV.")
                return None
            # take the last occurrence by default
            val = vals[-1]

        # Try direct float conversion first
        try:
            return float(val)
        except Exception:
            # handle strings like "(0.0,)" or "[0.0]"
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    return float(parsed[0])
                return float(parsed)
            except Exception:
                raise ValueError(f"Could not parse parameter '{parameter_name}' value: {val!r}")
    except IndexError:
        print(f"Error: Parameter '{parameter_name}' not found in CSV.")
        return None

# Extract your specific variables
t_climb = get_param('t_climb')             
t_cruise = get_param('t_cruise')           
t_reserve = get_param('t_reserve')         

P_climb = get_param('P_climb_shaft')       
P_cruise = get_param('P_cruise_shaft')

# Energy per flight phase that has to arrive at the shaft
E_climb = P_climb * t_climb
E_cruise = P_cruise * t_cruise
E_total = E_climb + E_cruise

# =============================================================================
lhv = 119930000 # J/kg, lower heating value of Hydrogen [standard property]
# =============================================================================

def load_efficiencies(comp):
    #design A: GT-BAT
    res_A = GT_BAT_efficiency(comp, t_climb=t_climb, t_cruise=t_cruise, P_climb=P_climb, P_cruise=P_cruise)
    A_gt_bt_eff = res_A["Total_eff"]
    A_P_gt = res_A.get("GT_P_opt")
    A_bt_eff_d = res_A.get("BAT-MOT_eff")
    A_cruise_eff_c = res_A.get("Cruise_charging_eff")
    A_gt_eff = res_A.get("LH2-GT-MOT_eff")

    #design B: FC-BAT
    res_B = FC_BAT_efficiency(comp, t_climb=t_climb, t_cruise=t_cruise, P_climb=P_climb, P_cruise=P_cruise)
    B_fc_bt_eff = res_B["Total_eff"]
    B_P_fc = res_B.get("FC_P")
    B_bt_eff_d = res_B.get("BAT-MOT_eff")
    B_cruise_eff_c = res_B.get("Cruise_charging_eff")
    B_cruise_eff_nc = res_B.get("Cruise_noncharging_eff")
    B_fc_eff = res_B.get("LH2-FC-MOT_eff")

    #design C: GT-FC
    res_C = GT_FC_efficiency(comp, t_climb=t_climb, t_cruise=t_cruise, P_climb=P_climb, P_cruise=P_cruise)
    C_gt_fc_eff = res_C["Total_eff"]
    C_P_gt_opt = res_C.get("GT_P_opt")
    C_throttle_climb = res_C.get("GT_throttle_climb")
    C_throttle_cruise = res_C.get("GT_throttle_cruise")
    C_P_gt = C_P_gt_opt * (C_throttle_cruise if C_throttle_cruise is not None else 1.0) if C_P_gt_opt is not None else None
    C_P_gt_climb = C_P_gt_opt * (C_throttle_climb if C_throttle_climb is not None else 1.0) if C_P_gt_opt is not None else None
    C_gt_eff = res_C.get("LH2-GT-MOT_eff")
    C_P_fc = res_C.get("FC_P")
    C_fc_eff = res_C.get("LH2-FC-MOT_eff")
    C_cruise_eff = res_C.get("Cruise_average_eff")

    #design D: GT-GT
    res_D = GT_GT_efficiency(comp, t_climb=t_climb, t_cruise=t_cruise, P_climb=P_climb, P_cruise=P_cruise)
    D_gt_gt_eff = res_D["Total_eff"]
    D_P_gt_climb = res_D.get("GT_P_opt") * res_D.get("GT_throttle_climb") if res_D.get("GT_P_opt") is not None else None
    D_climb_eff = res_D.get("Climb_eff")
    D_P_gt_cruise = res_D.get("GT_P_opt") * res_D.get("GT_throttle_cruise") if res_D.get("GT_P_opt") is not None else None
    D_cruise_eff = res_D.get("Cruise_average_eff")

    return {
        "A_gt_bt_eff": A_gt_bt_eff,
        "A_cruise_eff_c": A_cruise_eff_c,
        "A_gt_eff": A_gt_eff,
        "A_P_gt": A_P_gt,
        "A_bt_eff_d": A_bt_eff_d,
        "A_P_bt_discharge": res_A.get("BAT_P_discharge"),
        "B_fc_bt_eff": B_fc_bt_eff,
        "B_cruise_eff_c": B_cruise_eff_c,
        "B_cruise_eff_nc": B_cruise_eff_nc,
        "B_fc_eff": B_fc_eff,
        "B_P_fc": B_P_fc,
        "B_bt_eff_d": B_bt_eff_d,
        "B_P_bt_discharge": res_B.get("BAT_P_discharge"),
        "C_gt_fc_eff": C_gt_fc_eff,
        "C_cruise_eff": C_cruise_eff,
        "C_gt_eff": C_gt_eff,
        "C_P_gt_climb": C_P_gt_climb,
        "C_P_fc": C_P_fc,
        "C_fc_eff": C_fc_eff,
        "D_gt_gt_eff": D_gt_gt_eff,
        "D_cruise_eff": D_cruise_eff,
        "D_climb_eff": D_climb_eff,
        "D_P_gt_climb": D_P_gt_climb,
    }


def build_designs(eff):
    return {
        'GT-BAT': {
            'cruise': {'source': 'GT', 'eta': eff['A_cruise_eff_c']},
            'to_climb': {'primary': 'GT', 'eta_primary': eff['A_gt_eff'], 'p_primary': eff['A_P_gt'], 'secondary': 'BAT', 'eta_secondary': eff['A_bt_eff_d'], 'p_secondary': eff['A_P_bt_discharge']}
        },
        'FC-BAT': {
            'cruise': {'source': 'FC', 'eta': eff['B_cruise_eff_c']},
            'to_climb': {'primary': 'FC', 'eta_primary': eff['B_fc_eff'], 'p_primary': eff['B_P_fc'], 'secondary': 'BAT', 'eta_secondary': eff['B_bt_eff_d'], 'p_secondary': eff['B_P_bt_discharge']}
        },
        'GT-GT': {
            'cruise': {'source': 'GT', 'eta': eff['D_cruise_eff']},
            'to_climb': {'primary': 'GT', 'eta_primary': eff['D_climb_eff'], 'p_primary': eff['D_P_gt_climb'] / 2, 'secondary': 'GT2', 'eta_secondary': eff['D_climb_eff'], 'p_secondary': eff['D_P_gt_climb'] / 2}
        },
        'GT-FC': {
            'cruise': {'source': 'FC', 'eta': eff['C_cruise_eff']},
            'to_climb': {'primary': 'GT', 'eta_primary': eff['C_gt_eff'], 'p_primary': eff['C_P_gt_climb'], 'secondary': 'FC', 'eta_secondary': eff['C_fc_eff'], 'p_secondary': eff['C_P_fc']}
        },
        'Baseline': {
            'cruise': {'source': 'Jet', 'eta': 0.33},
            'to_climb': {'primary': 'Jet', 'eta_primary': 0.33, 'p_primary': eff['D_P_gt_climb'] / 2, 'secondary': 'Jet', 'eta_secondary': 0.33, 'p_secondary': eff['D_P_gt_climb'] / 2}
        },
        'GT-GT-BAT': {
            'cruise': {'source': 'GT', 'eta': eff['A_cruise_eff_c']},
            'to_climb': {'primary': 'GT', 'eta_primary': eff['A_gt_eff'], 'p_primary': eff['A_P_gt'], 'secondary': 'BAT', 'eta_secondary': eff['A_bt_eff_d'], 'p_secondary': eff['A_P_bt_discharge']}
        },
    }
source_props = {
    'GT': {'nox': True, 'h2o': True, 'contrail': True, 'co2': False},
    'GT2': {'nox': True, 'h2o': True, 'contrail': True, 'co2': False},
    'FC': {'nox': False, 'h2o': True, 'contrail': False, 'co2': False},
    'BAT': {'nox': False, 'h2o': False, 'contrail': False, 'co2': False},
    'Jet': {'nox': True, 'h2o': True, 'contrail': True, 'co2': True},
}

def calc_mass_h2(comp, efficiency_results=None):
    """Return hydrogen mass by design, phase, and source."""

    if efficiency_results is None:
        efficiency_results = load_efficiencies(comp)

    designs = build_designs(efficiency_results)
    A_gt_bt_eff = efficiency_results['A_gt_bt_eff']
    A_gt_eff = efficiency_results['A_gt_eff']
    A_P_gt = efficiency_results['A_P_gt']
    A_bt_eff_d = efficiency_results['A_bt_eff_d']
    B_fc_bt_eff = efficiency_results['B_fc_bt_eff']
    B_fc_eff = efficiency_results['B_fc_eff']
    B_P_fc = efficiency_results['B_P_fc']
    B_bt_eff_d = efficiency_results['B_bt_eff_d']
    C_gt_fc_eff = efficiency_results['C_gt_fc_eff']
    C_gt_eff = efficiency_results['C_gt_eff']
    C_P_gt_climb = efficiency_results['C_P_gt_climb']
    C_P_fc = efficiency_results['C_P_fc']
    C_fc_eff = efficiency_results['C_fc_eff']
    C_cruise_eff = efficiency_results['C_cruise_eff']
    D_gt_gt_eff = efficiency_results['D_gt_gt_eff']
    D_climb_eff = efficiency_results['D_climb_eff']
    D_P_gt_climb = efficiency_results['D_P_gt_climb']
    energies = {
        'cruise': E_cruise,
        'climb': E_climb,
    }

    h2_masses = {}

    for design_name, design in designs.items():
        h2_masses[design_name] = {}

        cruise_source = design['cruise']['source']
        cruise_eta = design['cruise']['eta']
        # fallback if cruise efficiency not provided by the efficiency helpers
        if cruise_eta is None:
            try:
                total_eff_map = {
                    'GT-BAT': A_gt_bt_eff,
                    'FC-BAT': B_fc_bt_eff,
                    'GT-GT': D_gt_gt_eff,
                    'GT-FC': C_gt_fc_eff,
                    'Baseline': 0.33,
                }
                cruise_eta = total_eff_map.get(design_name) or 0.33
            except Exception:
                cruise_eta = 0.33
        cruise_energy = energies['cruise']
        cruise_m_h2 = 0.0

        # Special handling for GT-FC: both FC and GT supply cruise power.
        if design_name == 'GT-FC':
            # Primary rule: FC runs at its available cruise power (FC_P) and GT supplies the remainder
            if C_P_fc is not None:
                p_fc = C_P_fc
                # ensure not negative remainder
                p_gt = max(0.0, P_cruise - p_fc)
                energy_fc = p_fc * t_cruise
                energy_gt = p_gt * t_cruise

                m_h2_fc = 0.0
                if C_fc_eff:
                    m_h2_fc = energy_fc / (C_fc_eff * lhv)

                m_h2_gt = 0.0
                if C_gt_eff and p_gt > 0:
                    m_h2_gt = energy_gt / (C_gt_eff * lhv)

                cruise_m_h2 = m_h2_fc + m_h2_gt
                h2_masses[design_name]['cruise'] = {
                    'source': 'FC+GT',
                    'energy_kJ': cruise_energy,
                    'eta': cruise_eta,
                    'm_h2_kg': cruise_m_h2,
                    'cruise_breakdown': [
                        {'source': 'FC', 'energy_kJ': energy_fc, 'eta': C_fc_eff, 'm_h2_kg': m_h2_fc},
                        {'source': 'GT', 'energy_kJ': energy_gt, 'eta': C_gt_eff, 'm_h2_kg': m_h2_gt},
                    ],
                }
                # Debug output for GT-FC split
                """ print("GT-FC cruise split debug:")
                print(f"  C_P_fc={C_P_fc}, derived C_P_gt={p_gt}")
                print(f"  energy_fc={energy_fc}, energy_gt={energy_gt}")
                print(f"  C_fc_eff={C_fc_eff}, C_gt_eff={C_gt_eff}")
                print(f"  m_h2_fc={m_h2_fc}, m_h2_gt={m_h2_gt}, total_cruise_m_h2={cruise_m_h2}") """
            else:
                # fallback: if FC_P not provided, fall back to previous power-share split when both provided
                p_fc = C_P_fc
                p_gt = C_P_gt
                if (p_fc is not None) and (p_gt is not None) and (p_fc + p_gt) > 0:
                    total_p = p_fc + p_gt
                    energy_fc = cruise_energy * (p_fc / total_p)
                    energy_gt = cruise_energy * (p_gt / total_p)

                    m_h2_fc = 0.0
                    if C_fc_eff:
                        m_h2_fc = energy_fc / (C_fc_eff * lhv)

                    m_h2_gt = 0.0
                    if C_gt_eff:
                        m_h2_gt = energy_gt / (C_gt_eff * lhv)

                    cruise_m_h2 = m_h2_fc + m_h2_gt
                    h2_masses[design_name]['cruise'] = {
                        'source': 'FC+GT',
                        'energy_kJ': cruise_energy,
                        'eta': cruise_eta,
                        'm_h2_kg': cruise_m_h2,
                        'cruise_breakdown': [
                            {'source': 'FC', 'energy_kJ': energy_fc, 'eta': C_fc_eff, 'm_h2_kg': m_h2_fc},
                            {'source': 'GT', 'energy_kJ': energy_gt, 'eta': C_gt_eff, 'm_h2_kg': m_h2_gt},
                        ],
                    }
                    """ print("GT-FC cruise split debug (fallback share):")
                    print(f"  C_P_fc={C_P_fc}, C_P_gt={C_P_gt}")
                    print(f"  energy_fc={energy_fc}, energy_gt={energy_gt}")
                    print(f"  C_fc_eff={C_fc_eff}, C_gt_eff={C_gt_eff}")
                    print(f"  m_h2_fc={m_h2_fc}, m_h2_gt={m_h2_gt}, total_cruise_m_h2={cruise_m_h2}") """
                else:
                    # last-resort fallback: treat cruise as single source
                    if cruise_source != 'BAT':
                        cruise_m_h2 = cruise_energy / (cruise_eta * lhv)
                    h2_masses[design_name]['cruise'] = {
                        'source': cruise_source,
                        'energy_kJ': cruise_energy,
                        'eta': cruise_eta,
                        'm_h2_kg': cruise_m_h2,
                    }
            
        else:
            if cruise_source != 'BAT':
                cruise_m_h2 = cruise_energy / (cruise_eta * lhv)

            h2_masses[design_name]['cruise'] = {
                'source': cruise_source,
                'energy_kJ': cruise_energy,
                'eta': cruise_eta,
                'm_h2_kg': cruise_m_h2,
            }

        to_climb = design['to_climb']
        primary_source = to_climb['primary']
        primary_eta = to_climb['eta_primary']
        if primary_eta is None:
            primary_eta = cruise_eta
        primary_power = to_climb['p_primary']
        if primary_power is None:
            print(f"Warning: 'p_primary' missing for design '{design_name}', treating as 0.0 W")
            primary_power = 0.0

        secondary_source = to_climb['secondary']
        secondary_eta = to_climb['eta_secondary']
        if secondary_eta is None:
            secondary_eta = cruise_eta
        secondary_power = to_climb['p_secondary']
        if secondary_power is None:
            print(f"Warning: 'p_secondary' missing for design '{design_name}', treating as 0.0 W")
            secondary_power = 0.0

        total_power = primary_power + secondary_power
        if total_power <= 0:
            raise ValueError(f"Total TO/climb power must be positive for design '{design_name}'.")

        climb_energy = energies['climb']
        primary_energy = climb_energy * (primary_power / total_power)
        secondary_energy = climb_energy * (secondary_power / total_power)

        primary_m_h2 = 0.0
        if primary_source != 'BAT':
            primary_m_h2 = primary_energy / (primary_eta * lhv)

        secondary_m_h2 = 0.0
        if secondary_source != 'BAT':
            secondary_m_h2 = secondary_energy / (secondary_eta * lhv)

        h2_masses[design_name]['to_climb'] = {
            'primary': {
                'source': primary_source,
                'power_W': primary_power,
                'energy_kJ': primary_energy,
                'eta': primary_eta,
                'm_h2_kg': primary_m_h2,
            },
            'secondary': {
                'source': secondary_source,
                'power_W': secondary_power,
                'energy_kJ': secondary_energy,
                'eta': secondary_eta,
                'm_h2_kg': secondary_m_h2,
            },
            'energy_split': {
                'primary_fraction': primary_power / total_power,
                'secondary_fraction': secondary_power / total_power,
            },
        }

        h2_masses[design_name]['total_m_h2_kg'] = cruise_m_h2 + primary_m_h2 + secondary_m_h2

    return h2_masses


def print_mass_h2_summary(h2_masses):
    print("\nHydrogen mass summary:")
    for design_name, design_data in h2_masses.items():
        cruise = design_data['cruise']
        primary = design_data['to_climb']['primary']
        secondary = design_data['to_climb']['secondary']
        total = design_data['total_m_h2_kg']

        """ print(f"\n{design_name}")
        print(f"  Cruise    ({cruise['source']}):     {cruise['m_h2_kg']:.3f} kg H2")
        print(f"  TO/Climb P ({primary['source']}):     {primary['m_h2_kg']:.3f} kg H2")
        print(f"  TO/Climb S ({secondary['source']}):   {secondary['m_h2_kg']:.3f} kg H2")
        print(f"  Total:                    {total:.3f} kg H2") """

def calc_aCCF_nox():

    d = -23.44*cos(radians(360/365*(n+10)))
    F_in = s*(sin(radians(latitude))*sin(radians(d)) + cos(radians(latitude))*cos(radians(d)))

    aCCF_o3 = -2.64*10**(-11) + 1.17*10**(-13)*t + 2.46*10**(-16)*geopotential - 1.04*(10**-18)*t*geopotential    
    aCCF_ch4 = -4.84*10**(-13) + 9.79*10**(-19)*geopotential - 3.11*10**(-16)*F_in + 3.01*10**(-21)*F_in*geopotential
    aCCF_pmo = 0.29 * aCCF_ch4

    # print(f"NOx impact: {aCCF_o3 + aCCF_ch4 + aCCF_pmo}")
    return aCCF_o3 + aCCF_ch4 + aCCF_pmo

def calc_aCCF_h2o():

    aCCF_h2o = (2.11*10**(-16) + 7.70*10**(-17)*abs(pv))*(9/1.231)

    # print(f"H2O impact: {aCCF_h2o}")
    return aCCF_h2o

def calc_aCCF_contrail():

    if t > 201:
        aCCF_contrail_night = 0.0151*((10**(-10))*(0.0073*10**(0.0107*t)-1.03))
    else:
        aCCF_contrail_night = 0
    aCCF_contrail_day = 0.0151*((10**(-10))*(-1.7-0.0088*olr))
    
    aCCF_contrail_mean = f_day*aCCF_contrail_day + (1-f_day)*aCCF_contrail_night

    # print(f"Contrail impact: {aCCF_contrail_mean}")
    return aCCF_contrail_mean

def calc_aCCF_co2():
    aCCF_co2 = 7.48*10**(-16)

    #  print(f"CO2 impact: {aCCF_co2}")
    return aCCF_co2

def calc_atr_per_design(h2_masses):
    aCCF_nox = calc_aCCF_nox()
    aCCF_h2o = calc_aCCF_h2o()
    aCCF_contrail = calc_aCCF_contrail()
    aCCF_co2 = calc_aCCF_co2()

    atrs = {}
    for design_name, design_data in h2_masses.items():
        cruise = design_data['cruise']
        primary = design_data['to_climb']['primary']
        secondary = design_data['to_climb']['secondary']

        if design_name == 'Jet': # only apply energy density adjustment for jet
            f_ed = f_energy_density
        else:
            f_ed = 1.0
        
        atr_cruise = 0.0
        cruise_nox = cruise_h2o = cruise_co2 = cruise_contrail = 0.0
        # If cruise has a detailed breakdown, attribute impacts per contributing source
        cruise_breakdown = cruise.get('cruise_breakdown') if isinstance(cruise, dict) else None
        if cruise_breakdown:
            # compute per-source NOx/H2O/CO2 then add a single contrail term
            total_contrail_mass = sum(part.get('m_h2_kg', 0.0) for part in cruise_breakdown if source_props[part['source']]['contrail'])
            total_cruise_mass = sum(part.get('m_h2_kg', 0.0) for part in cruise_breakdown)
            for part in cruise_breakdown:
                src = part['source']
                m_h2 = part.get('m_h2_kg', 0.0)
                if source_props[src]['nox']:
                    val = m_h2 * ei_nox * aCCF_nox * f_ed
                    atr_cruise += val
                    cruise_nox += val
                if source_props[src]['h2o']:
                    val = m_h2 * aCCF_h2o * f_ed
                    atr_cruise += val
                    cruise_h2o += val
                if source_props[src]['co2']:
                    val = m_h2 * aCCF_co2 * f_energy_density
                    atr_cruise += val
                    cruise_co2 += val
            # add contrail once, scaled by fraction of cruise mass from contrail-producing sources
            if total_cruise_mass > 0 and total_contrail_mass > 0:
                contrail_total = d_mission * f_ISSR * aCCF_contrail * (total_contrail_mass / total_cruise_mass)
                atr_cruise += contrail_total
                cruise_contrail += contrail_total
        else:
            cruise_nox = cruise_h2o = cruise_co2 = cruise_contrail = 0.0
            if source_props[cruise['source']]['nox']:
                cruise_nox = cruise['m_h2_kg'] * ei_nox * aCCF_nox * f_ed
                atr_cruise += cruise_nox
            if source_props[cruise['source']]['h2o']:
                cruise_h2o = cruise['m_h2_kg'] * aCCF_h2o * f_ed
                atr_cruise += cruise_h2o
            if source_props[cruise['source']]['contrail']:
                cruise_contrail = d_mission * f_ISSR * aCCF_contrail
                atr_cruise += cruise_contrail
            if source_props[cruise['source']]['co2']:
                cruise_co2 = cruise['m_h2_kg'] * aCCF_co2 * f_energy_density # adjust for energy density difference to jet fuel
                atr_cruise += cruise_co2

        atr_primary = 0.0
        if source_props[primary['source']]['nox']:
            atr_primary += primary['m_h2_kg'] * ei_nox * aCCF_nox * f_ed
        if source_props[primary['source']]['h2o']:
            atr_primary += primary['m_h2_kg'] * aCCF_h2o * f_ed
        if source_props[primary['source']]['co2']:
            atr_primary += primary['m_h2_kg'] * aCCF_co2 * f_energy_density 
            
        atr_secondary = 0.0
        sec_nox = sec_h2o = sec_co2 = 0.0
        if source_props[secondary['source']]['nox']:
            sec_nox = secondary['m_h2_kg'] * ei_nox * aCCF_nox * f_ed
            atr_secondary += sec_nox
        if source_props[secondary['source']]['h2o']:
            sec_h2o = secondary['m_h2_kg'] * aCCF_h2o * f_ed
            atr_secondary += sec_h2o
        if source_props[secondary['source']]['co2']:
            sec_co2 = secondary['m_h2_kg'] * aCCF_co2 * f_energy_density
            atr_secondary += sec_co2

        total_atr = atr_cruise + atr_primary + atr_secondary
        atrs[design_name] = total_atr

    """ # Debug print: detailed ATR contributions
        print(f"ATR breakdown for {design_name}:")
        print(f"  Cruise total={atr_cruise} (NOx={cruise_nox}, H2O={cruise_h2o}, CO2={cruise_co2}, Contrail={cruise_contrail})")
        print(f"  TO/Climb primary={atr_primary}")
        print(f"  TO/Climb secondary={atr_secondary}")
        print(f"  Total ATR={total_atr}\n")
 """
    #print(atrs)
    return atrs


def get_results(comp=None):
    """Return the ATR results dictionary for all designs."""
    if comp is None:
        comp = component_params
    return calc_atr_per_design(calc_mass_h2(comp))



if __name__ == "__main__":
    print(get_results())
    

