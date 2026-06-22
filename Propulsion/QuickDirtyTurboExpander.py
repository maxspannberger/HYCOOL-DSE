import numpy as np
from CoolProp.CoolProp import PropsSI

Psi                 = 1.5                   # Work Coefficient per stage for Radial Turboexpander
T_Hydrogen_In       = 850                   # K
P_Hydrogen_In       = 150                   # bar
P_Hydrogen_Out      = 10                    # bar
mdot_Hydrogen       = 0.03                  # kg/s
eta_Turboexp        = 0.88                  # Assumed Isentropic efficiency
fluid               = 'ParaHydrogen'


h_in                = PropsSI("H", "P", P_Hydrogen_In*1e5, "T", T_Hydrogen_In, fluid)
s_in                = PropsSI("S", "P", P_Hydrogen_In*1e5, "T", T_Hydrogen_In, fluid)
h_out_Isentropic    = PropsSI("H", "P", P_Hydrogen_Out*1e5, "S", s_in, fluid)
h_out_real          = h_in - (h_in - h_out_Isentropic) * eta_Turboexp

delta_h             = h_in - h_out_real


sigma_max           = 1400e6                 # Pa, maximum stress for material (IN718)
rho_material        = 8190                  # kg/m3
shape_factor        = 0.45                  # between 0.3-0.6 according to clanker

U_1_max             = np.sqrt(sigma_max / (shape_factor * rho_material))                   # m/s, maximum tip speed, material limit
delta_h_stage_max   = Psi * U_1_max**2      # Maximum change in enthalpy per stage

N_stages            = int(np.ceil(delta_h / delta_h_stage_max))

delta_h_stage       = delta_h / N_stages 
U1                  = np.sqrt(delta_h_stage / Psi)

print(f"\nNumber of stages: {N_stages}")
print(f"Maximum allowed tip speed: {U_1_max:.1f}")
print(f"Enthalpy Change: {delta_h:.0f}")
print(f"Enthalpy Changer per Stage: {delta_h_stage:.0f}")
print(f"Power Extracted: {delta_h * mdot_Hydrogen/1e3:.2f} kW")
print(f"Real Tip Velocity: {U1:.1f}\n")