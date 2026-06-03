# Import statements
from CoolProp import AbstractState
from CoolProp.CoolProp import PhaseSI, PropsSI, get_global_param_string
import CoolProp.CoolProp as CoolProp
from CoolProp.HumidAirProp import HAPropsSI
import numpy as np

# Tank Insulation Properties
rhoTankInsulation = 200.0 # kg/m^3 [PolyeuretheneFoam]

# Tank Wall Material Properties
rhoTankWall = 2825.0 # kg/m^3 [AL-2219]


# Crash Worthiness Parameters
beta = None # Crash diameter coefficient
AR_f = None # Aspect ratio of the fuselage at the limiting section

# Tank Design Parameters
tauH = 120 # min, holding time for liquid hydrogen 

# Atmospheric Properties
BAR = 100000 # Pa, 1 bar in SI units

# Hydrogen Parameters
TLH2 = 20.0 # K, temperature of liquid hydrogen in the tank
pLH2 = 1 * BAR # Pa, absolute pressure used for saturation properties
rhoLH2Saturated = PropsSI('D', 'P', pLH2, 'Q', 0, 'parahydrogen')   # Q=0: saturated liquid
rhoGH2Saturated = PropsSI('D', 'P', pLH2, 'Q', 1, 'parahydrogen')   # Q=1: saturated vapor
vl0 = 1/rhoLH2Saturated # m^3/kg, specific volume of liquid hydrogen at saturation
vg0 = 1/rhoGH2Saturated # m^3/kg, specific volume of gaseous hydrogen at saturation
