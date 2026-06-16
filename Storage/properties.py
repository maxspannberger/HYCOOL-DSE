from CoolProp.CoolProp import PropsSI
from scipy.integrate import quad

bar = 100000 # Pa, 1 bar in SI units
FLUID = "parahydrogen"   # physically correct LH2 near 20 K; 'hydrogen' = normal H2

# Calculate the specific volumes of the saturated liquid and vapor at a given pressure
def saturated_specific_volumes(P, fluid):
    vl_0 = 1.0/PropsSI("D","P",P,"Q",0,fluid)
    vg_0 = 1.0/PropsSI("D","P",P,"Q",1,fluid)
    return vl_0, vg_0

# Calculate the saturated mean density of the liquid hydrogen
def saturated_densities(P, fluid):
    """Calculate the density of the mixture at a given pressure (in bar) and initial liquid mass fraction."""
    rho_H2 = PropsSI('D', 'P', P * bar, 'Q', 0, fluid)
    return rho_H2

# Calculate the specific internal energies of saturated liquid and vapor at a given pressure (Pa)
def saturated_internal_energies(P, fluid):
    """Return specific internal energies (J/kg) of saturated liquid and vapour at pressure P (Pa)."""
    ul = PropsSI('U', 'P', P, 'Q', 0, fluid)
    ug = PropsSI('U', 'P', P, 'Q', 1, fluid)
    return ul, ug
