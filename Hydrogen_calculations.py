from CoolProp import AbstractState
from CoolProp.CoolProp import PhaseSI, PropsSI, get_global_param_string
import CoolProp.CoolProp as CoolProp
from CoolProp.HumidAirProp import HAPropsSI

fluid = 'hydrogen'
# See http://www.coolprop.org/coolprop/HighLevelAPI.html#table-of-string-inputs-to-propssi-function for a list of inputs to high-level interface
print("*********** HIGH LEVEL INTERFACE *****************")
print("Critical temperature of hydrogen:", PropsSI("Tcrit", fluid), "K")
print("Boiling temperature of hydrogen at 101325 Pa:", PropsSI("T", "P", 101325, "Q", 0, fluid), "K")
print("Phase of hydrogen at 101325 Pa and 300 K:", PhaseSI("P", 101325, "T", 300, fluid))
print("c_p of hydrogen at 101325 Pa and 300 K:", PropsSI("C", "P", 101325, "T", 300, fluid), "J/kg/K")
print("c_p of hydrogen (using derivatives) at 101325 Pa and 300 K:", PropsSI("d(H)/d(T)|P", "P", 101325, "T", 300, fluid), "J/kg/K")
print("*********** HUMID AIR PROPERTIES *****************")
print("Humidity ratio of 50% rel. hum. air at 300 K, 101325 Pa:", HAPropsSI("W", "T", 300, "P", 101325, "R", 0.5), "kg_w/kg_da")
print("Relative humidity from last calculation:", HAPropsSI("R", "T", 300, "P", 101325, "W", HAPropsSI("W", "T", 300, "P", 101325, "R", 0.5)), "(fractional)")
print("*********** INCOMPRESSIBLE FLUID AND BRINES *****************")
print("Density of 50% (mass) ethylene glycol/water at 300 K, 101325 Pa:", PropsSI("D", "T", 300, "P", 101325, "INCOMP::MEG-50%"), "kg/m^3")
print("Viscosity of Therminol D12 at 350 K, 101325 Pa:", PropsSI("V", "T", 350, "P", 101325, "INCOMP::TD12"), "Pa-s")


print("*********** TABULAR BACKENDS *****************")
TAB = AbstractState("BICUBIC&HEOS", "R245fa")
TAB.update(CoolProp.PT_INPUTS, 101325, 300)
print("Mass density of refrigerant R245fa at 300 K, 101325 Pa:", TAB.rhomass(), "kg/m^3")
print("*********** SATURATION DERIVATIVES (LOW-LEVEL INTERFACE) ***************")
AS_SAT = AbstractState("HEOS", "R245fa")
AS_SAT.update(CoolProp.PQ_INPUTS, 101325, 0)
print("First saturation derivative:", AS_SAT.first_saturation_deriv(CoolProp.iP, CoolProp.iT), "Pa/K")

