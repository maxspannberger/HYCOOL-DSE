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

lhv = 119930 # kJ/kg, lower heating value of Hydrogen [standard property - citation TBD]
energies = {
    'cruise': 500000 #kJ,
    'climb': 100000 #kJ, 
}
designs = {
    'GT-BAT': {
        'cruise': {'source': 'GT', 'eta': 0.35},
        'to_climb': {'primary': 'GT', 'eta_primary': 0.35, 'p_primary': 500, 'secondary': 'BAT', 'eta_secondary': 0.4, 'p_secondary': 300}
    },
    'FC-BAT':{
        'cruise': {'source': 'FC', 'eta': 0.35},
        'to_climb': {'primary': 'FC', 'eta_primary': 0.35, 'p_primary': 500, 'secondary': 'BAT', 'eta_secondary': 0.4, 'p_secondary': 300}
    },
    'GT-GT':{
        'cruise': {'source': 'GT', 'eta': 0.35},
        'to_climb': {'primary': 'GT', 'eta_primary': 0.35, 'p_primary': 500, 'secondary': 'GT2', 'eta_secondary': 0.4, 'p_secondary': 300}
    },
    'GT-FC':{
        'cruise': {'source': 'GT', 'eta': 0.35},
        'to_climb': {'primary': 'GT', 'eta_primary': 0.35, 'p_primary': 500, 'secondary': 'FC', 'eta_secondary': 0.4, 'p_secondary': 300}
    },
}
source_props = {
    'GT': {'nox': True, 'h2o': True, 'contrail': True},
    'GT2': {'nox': True, 'h2o': True, 'contrail': True},
    'FC': {'nox': False, 'h2o': True, 'contrail': False},
    'BAT': {'nox': False, 'h2o': False, 'contrail': False},
}

def calc_mass_h2():
    m_h2 = 0
    return m_h2

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


def calc_average_temperature_response():

    if turbine:
        aCCF_nox = calc_aCCF_nox()
        aCCF_h2o = calc_aCCF_h2o()
        aCCF_contrail = calc_aCCF_contrail()
    elif fc_liquid_venting:
        aCCF_nox = 0
        aCCF_h2o = calc_aCCF_h2o()
        aCCF_contrail = 0
    else:
        aCCF_nox = 0
        aCCF_h2o = 0
        aCCF_contrail = 0

    atr20 = aCCF_nox*m_h2*ei_nox + aCCF_h2o*m_h2 + aCCF_contrail*d_mission*f_ISSR

    return atr20


if __name__ == "__main__":
    atr20 = calc_average_temperature_response()
    print(f"Average Temperature Response (ATR20) for the mission: {atr20:.12f} K")

