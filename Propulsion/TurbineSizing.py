from rocketcea.cea_obj_w_units import CEA_Obj
import numpy as np
from rocketcea.blends import newFuelBlend
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

'''
Station Definitions:

P1 / T1: Pre-HPC
P2 / T2: Post-HPC / Pre CC
P3 / T3: CC
P4 / T4: Post CC / Pre HPT
P5 / T5: Post HPT

'''


# Todo: Add isentropic efficiencies

mdot_f          = 0.155                     # Fuel mass-flow rate. Should be iterated on

P_ambient       = 0.38                      # bar, 25,000ft

PR_HPC          = 30                        # Pressure ratio for the HP Compressor
eta_HPC         = 0.88                      # Isentropic efficiency for HPC. ADD

Pc              = P_ambient * PR_HPC        # Combustion pressure
eta_CC          = 1.0                       # Combustive efficiency for CC. ADD
eta_CC_p        = 1.0                       # Pressure drop across combustor

PR_HPT          = 11                        # Pressure ratio for the HP Turbine
eta_HPT         = 0.92                      # Isentropic efficiency for HPT. ADD

cea = CEA_Obj(oxName = "AIR", fuelName = "GH2", pressure_units = "bar", temperature_units = "K", isp_units = "sec")
of_ratios = np.linspace(1.0, 100.0, 50)
tc_vals = []

for of in of_ratios:

    tc_kelvin = cea.get_Tcomb(Pc=Pc, MR=of)
    tc_vals.append(tc_kelvin)

tc_vals = np.array(tc_vals)
peak_temp           = np.max(tc_vals)
peak_temp_index     = np.argmax(tc_vals)
of_stoic            = of_ratios[peak_temp_index]
stoic_params        = cea.get_exit_MolWt_gamma(Pc=Pc, MR=of_stoic, eps=1, frozen=1)
gamma_stoic         = stoic_params[1]
Rs_stoic            = 8.314 / (stoic_params[0] / 1000)

# USE THESE FOR CALCULATIONS
ideal_temp          = 1500              # K
ideal_OF            = of_ratios[np.abs(tc_vals - ideal_temp).argmin()]
ideal_params        = cea.get_exit_MolWt_gamma(Pc=Pc, MR=ideal_OF, eps=1, frozen=1)
Rs_ideal            = 8.314 / (ideal_params[0] / 1000)
mdot_tot            = mdot_f * (ideal_OF+1)
gamma_ideal         = PropsSI('CPMASS','P',Pc*1e5,'T',ideal_temp,'Air') / PropsSI('CVMASS','P',Pc*1e5,'T',ideal_temp,'Air')

print(f"\nStoichiometric Temperature: {peak_temp:.2f} K")
print(f"Stoichiometric O/F Ratio: {of_stoic:.2f}")
print(f"Stoichiometric Gamma: {gamma_stoic:.2f}")
print(f"Stoichiometric Specific Gas Constant: {Rs_stoic:.2f}\n ")

print(f"Ideal Temperature: {ideal_temp:.2f} K")
print(f"Ideal O/F Ratio: {ideal_OF:.2f}")
print(f"Ideal Gamma: {gamma_ideal:.2f}")
print(f"Ideal Specific Gas Constant: {Rs_ideal:.2f}\n")


gamma = gamma_ideal
Rs = Rs_ideal

T1 = 250
P1 = P_ambient

gamma_HPC = PropsSI('CPMASS', 'P', P1*1e5, 'T', T1, 'Air') / PropsSI('CVMASS', 'P', P1*1e5, 'T', T1, 'Air')
T2s = T1 * (PR_HPC) ** ((gamma_HPC - 1)/gamma_HPC)
P2 = P1 * PR_HPC

T12_Delta = (T2s - T1) / eta_HPC
T2 = T1 + T12_Delta
Cp_HPC = PropsSI('CPMASS', 'P', P2*1e5, 'T', 0.5*(T2+T1), 'Air')
P_HPC = Cp_HPC * mdot_tot * (T2 - T1)

T3 = ideal_temp
P3 = Pc

T4 = T3
P4 = P3 * eta_CC_p

T5s = T4 * (1 / PR_HPT)**((gamma - 1)/gamma)            # Isentropic, ideal temperature
P5 = Pc / PR_HPT

T45_Delta = (T4 - T5s) * eta_HPT
T5 = T4 - T45_Delta                                     # True temperature w/ efficiency
Cp_HPT = PropsSI('CPMASS', 'P', P5*1e5, 'T', 0.5*(T4+T5), 'Air')   # J/kg-K (SI, not kJ)

P_HPT = Cp_HPT * mdot_tot * (T4 - T5)                  # HPT Power output


print(f"HPC Input power: {P_HPC/1e6:.3f} MW")
print(f"Cp_HPC: {Cp_HPC:.2f} J/kg-K")
print(f"HPT Output power: {P_HPT/1e6:.3f} MW")
print(f"Cp_HPT: {Cp_HPT:.2f} J/kg-K")
print(f"Net Power Output: {(P_HPT-P_HPC)/1e6:.3f} MW")



# fig, ax = plt.subplots(figsize=(10, 6))

# color = 'tab:red'
# ax.set_ylabel('Combustion Temperature (K)', color=color)
# ax.plot(of_ratios, tc_vals, color=color, linewidth=2, linestyle='--', label='Tc')
# ax.tick_params(axis='y', labelcolor=color)

# fig.suptitle('Air/GH2: O/F Ratio vs Isp and Combustion Temperature')
# fig.tight_layout()
# plt.show()


# Combustion chamber parameters:
# TIT Gamma / R / Temperature (T4)
# Need T3 / Inlet parameters -> After HPC
# Look first at sizing HPT
# O/F 80, Mdot 155g/s -> Total Mdot = 12.56kg/s
# Entire flow of 12.56kg/s at 1500K, ideally want to extract all energy from this
# I guess a big pressure drop then? Try to fully expand from 11 bar combustion pressure to 1 bar, any extra pressure difference from alt is just more thrust. 
# => HPT Pressure ratio = 11, Temperature ratio =