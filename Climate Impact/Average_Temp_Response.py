import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd

from Propulsion.efficiency import (
    FC_BAT_efficiency,
    GT_BAT_efficiency,
    GT_FC_efficiency,
    GT_GT_efficiency,
)

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
def get_param(parameter_name):
    try:
        # We look for the parameter name and return the associated value
        val = df.loc[df['Parameter'] == parameter_name, 'Value'].values[0]
        return float(val)
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
# Loading results from  efficiency calculations 
# =============================================================================

#design A: GT-BAT
res_A = GT_BAT_efficiency()
A_gt_bt_eff = res_A["Total_eff"]
A_P_gt = res_A.get("GT_P_opt")
A_climb_eff = res_A.get("Climb_eff")
A_P_bt_discharge = res_A.get("BAT_P_discharge")
A_bt_eff_d = res_A.get("BAT-MOT_eff")
A_cruise_eff_c = res_A.get("Cruise_charging_eff")
A_gt_eff = res_A.get("LH2-GT-MOT_eff")

#design B: FC-BAT
res_B = FC_BAT_efficiency()
B_fc_bt_eff = res_B["Total_eff"]
B_P_fc = res_B.get("FC_P")
B_climb_eff = res_B.get("Climb_eff")
B_P_bt_discharge = res_B.get("BAT_P_discharge")
B_bt_eff_d = res_B.get("BAT-MOT_eff")
B_cruise_eff_c = res_B.get("Cruise_charging_eff")
B_fc_eff = res_B.get("LH2-FC-MOT_eff")

#design C: GT-FC
res_C = GT_FC_efficiency()
C_gt_fc_eff = res_C["Total_eff"]
C_P_gt = res_C.get("GT_P")
C_gt_eff = res_C.get("LH2-GT-MOT_eff")
C_P_fc = res_C.get("FC_P")
C_fc_eff = res_C.get("LH2-FC-MOT_eff")

#design D: GT-GT
res_D = GT_GT_efficiency()
D_gt_gt_eff = res_D["Total_eff"]
# compute per-engine climb/cruise powers from returned optimal and throttles
D_P_gt_climb = res_D.get("GT_P_opt") * res_D.get("GT_throttle_climb") if res_D.get("GT_P_opt") is not None else None
D_climb_eff = res_D.get("Climb_eff")
D_P_gt_cruise = res_D.get("GT_P_opt") * res_D.get("GT_throttle_cruise") if res_D.get("GT_P_opt") is not None else None
D_cruise_eff = res_D.get("Cruise_eff")

# =============================================================================

lhv = 119930000 # J/kg, lower heating value of Hydrogen [standard property]
energies = {
    'cruise': E_cruise ,
    'climb': E_climb , 
}
designs = {
    'GT-BAT': {
        'cruise': {'source': 'GT', 'eta': A_cruise_eff_c},
        'to_climb': {'primary': 'GT', 'eta_primary': A_gt_eff, 'p_primary': A_P_gt, 'secondary': 'BAT', 'eta_secondary': A_bt_eff_d, 'p_secondary': A_P_bt_discharge}
    },
    'FC-BAT':{
        'cruise': {'source': 'FC', 'eta': B_cruise_eff_c},
        'to_climb': {'primary': 'FC', 'eta_primary': B_fc_eff, 'p_primary': B_P_fc, 'secondary': 'BAT', 'eta_secondary': B_bt_eff_d, 'p_secondary': B_P_bt_discharge}
    },
    'GT-GT':{
        'cruise': {'source': 'GT', 'eta': D_cruise_eff},
        'to_climb': {'primary': 'GT', 'eta_primary': D_climb_eff, 'p_primary': D_P_gt_climb/2, 'secondary': 'GT2', 'eta_secondary': D_climb_eff, 'p_secondary': D_P_gt_climb/2}
    },
    'GT-FC':{
        'cruise': {'source': 'GT', 'eta': C_gt_eff},
        'to_climb': {'primary': 'GT', 'eta_primary': C_gt_eff, 'p_primary': C_P_gt, 'secondary': 'FC', 'eta_secondary': C_fc_eff, 'p_secondary': C_P_fc}
    },
    'Baseline':{
        'cruise': {'source': 'Jet', 'eta': 0.33},
        'to_climb': {'primary': 'Jet', 'eta_primary': 0.33, 'p_primary': D_P_gt_climb/2, 'secondary': 'Jet', 'eta_secondary': 0.33, 'p_secondary': D_P_gt_climb/2}
    },
}
source_props = {
    'GT': {'nox': True, 'h2o': True, 'contrail': True, 'co2': False},
    'GT2': {'nox': True, 'h2o': True, 'contrail': True, 'co2': False},
    'FC': {'nox': False, 'h2o': True, 'contrail': False, 'co2': False},
    'BAT': {'nox': False, 'h2o': False, 'contrail': False, 'co2': False},
    'Jet': {'nox': True, 'h2o': True, 'contrail': True, 'co2': True},
}

def calc_mass_h2():
    """Return hydrogen mass by design, phase, and source."""

    h2_masses = {}

    for design_name, design in designs.items():
        h2_masses[design_name] = {}

        cruise_source = design['cruise']['source']
        cruise_eta = design['cruise']['eta']
        cruise_energy = energies['cruise']
        cruise_m_h2 = 0.0
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
        primary_power = to_climb['p_primary']

        secondary_source = to_climb['secondary']
        secondary_eta = to_climb['eta_secondary']
        secondary_power = to_climb['p_secondary']

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

        print(f"\n{design_name}")
        print(f"  Cruise    ({cruise['source']}):     {cruise['m_h2_kg']:.3f} kg H2")
        print(f"  TO/Climb P ({primary['source']}):     {primary['m_h2_kg']:.3f} kg H2")
        print(f"  TO/Climb S ({secondary['source']}):   {secondary['m_h2_kg']:.3f} kg H2")
        print(f"  Total:                    {total:.3f} kg H2")

def calc_aCCF_nox():

    d = -23.44*cos(radians(360/365*(n+10)))
    F_in = s*(sin(radians(latitude))*sin(radians(d)) + cos(radians(latitude))*cos(radians(d)))

    aCCF_o3 = -2.64*10**(-11) + 1.17*10**(-13)*t + 2.46*10**(-16)*geopotential - 1.04*(10**-18)*t*geopotential    
    aCCF_ch4 = -4.84*10**(-13) + 9.79*10**(-19)*geopotential - 3.11*10**(-16)*F_in + 3.01*10**(-21)*F_in*geopotential
    aCCF_pmo = 0.29 * aCCF_ch4

    print(f"NOx impact: {aCCF_o3 + aCCF_ch4 + aCCF_pmo}")
    return aCCF_o3 + aCCF_ch4 + aCCF_pmo

def calc_aCCF_h2o():

    aCCF_h2o = (2.11*10**(-16) + 7.70*10**(-17)*abs(pv))*(9/1.231)

    print(f"H2O impact: {aCCF_h2o}")
    return aCCF_h2o

def calc_aCCF_contrail():

    if t > 201:
        aCCF_contrail_night = 0.0151*((10**(-10))*(0.0073*10**(0.0107*t)-1.03))
    else:
        aCCF_contrail_night = 0
    aCCF_contrail_day = 0.0151*((10**(-10))*(-1.7-0.0088*olr))
    
    aCCF_contrail_mean = f_day*aCCF_contrail_day + (1-f_day)*aCCF_contrail_night

    print(f"Contrail impact: {aCCF_contrail_mean}")
    return aCCF_contrail_mean

def calc_aCCF_co2():
    aCCF_co2 = 7.48*10**(-16)

    print(f"CO2 impact: {aCCF_co2}")
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
        if source_props[cruise['source']]['nox']:
            atr_cruise += cruise['m_h2_kg'] * ei_nox * aCCF_nox * f_ed
        if source_props[cruise['source']]['h2o']:
            atr_cruise += cruise['m_h2_kg'] * aCCF_h2o * f_ed
        if source_props[cruise['source']]['contrail']:
            atr_cruise += d_mission * f_ISSR * aCCF_contrail
        if source_props[cruise['source']]['co2']:
            atr_cruise += cruise['m_h2_kg'] * aCCF_co2 * f_energy_density # adjust for energy density difference to jet fuel

        atr_primary = 0.0
        if source_props[primary['source']]['nox']:
            atr_primary += primary['m_h2_kg'] * ei_nox * aCCF_nox * f_ed
        if source_props[primary['source']]['h2o']:
            atr_primary += primary['m_h2_kg'] * aCCF_h2o * f_ed
        if source_props[primary['source']]['co2']:
            atr_primary += primary['m_h2_kg'] * aCCF_co2 * f_energy_density 
            
        atr_secondary = 0.0
        if source_props[secondary['source']]['nox']:
            atr_secondary += secondary['m_h2_kg'] * ei_nox * aCCF_nox * f_ed
        if source_props[secondary['source']]['h2o']:
            atr_secondary += secondary['m_h2_kg'] * aCCF_h2o * f_ed
        if source_props[secondary['source']]['co2']:
            atr_secondary += secondary['m_h2_kg'] * aCCF_co2 * f_energy_density

        total_atr = atr_cruise + atr_primary + atr_secondary
        atrs[design_name] = total_atr

    print(atrs)
    return atrs


if __name__ == "__main__":
    h2_masses = calc_mass_h2()
    print_mass_h2_summary(h2_masses)

    calc_atr_per_design(h2_masses)

