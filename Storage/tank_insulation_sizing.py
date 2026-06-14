"""
Size the MLI insulation layer of the LH2 tank.
"""
import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq
from tank_size_shape import Geometry
import properties as props

#vl_vent = props.saturated_specific_volumes(self.p_vent * self.BAR, FLUID)[0] 
# Filling condition
mLH2 = 600
Pfill = 1.75e5
Pvent = 1.5 * Pfill
rhoLH2 = 70
Ef = PropsSI('U', 'P', Pfill , 'D', rhoLH2, 'parahydrogen')
Ei = PropsSI('U', 'P', Pvent , 'D', rhoLH2, 'parahydrogen')
tauH = 24 * 3600

# Tank geometry
ls = 2.0 # Geometry.ls
twall = 0.004
Rtank = 0.8 # Geometry.a
Atank = 4 * np.pi * (Rtank ** 2) + 2 * np.pi * Rtank * ls

SF = 1
Qleak = mLH2 * (Ei - Ef) / tauH
Qtank = Qleak / SF

# Temperature
Ts = 298  # [K]
Tc = 20.3  # [K]
Tm = (Ts - Tc) / 2 # [K]

# Lockheed MLI Parameters
P = 1e-4  # [torr]
Cs = 7.30e-8# 2.4e-7 * (0.017 + 7e-6 * (800 - Tm) + 0.0228 * np.log(Tm))
Cg = 1.46e4
Cr = 7.07e-10# 4.944e-10
emissivity =  0.043
rhoMLI = 20  # [kg/m^2]


# Calculate Nbar
Nbar = (( (Cr * emissivity * ((Ts ** 4.67) - (Tc ** 4.67)) + Cg * P * ((Ts ** 0.52) - (Tc ** 0.52))) ) / (0.78 * Cs * ((Ts ** 2) - (Tc ** 2)))) ** (1 / 2.63)
# Calculate qleak and Aout as functions of the number of MLI layers N
def qleak_func():
    qs = (Cs * (Nbar ** 2.63) * ((Ts ** 2) - (Tc ** 2))) / 2 
    qr = (Cr * emissivity * ((Ts ** 4.67) - (Tc ** 4.67))) / 1
    qg = Cg * P * ((Ts ** 0.52) - (Tc ** 0.52)) / 1
    return qs + qr + qg




def residual(N):
    return Qtank - qleak_func(N) * Atank


# Solve Qtank = qleak * Aout for the number of MLI layers N
qleak = qleak_func()
qtank = Qleak / Atank
N = np.ceil(qleak / qtank)
qMLI = qleak / N
tMLI = (N / Nbar) * 0.01    # Nbar is in layers/cm, convert thickness to [m]
Aout = 4 * np.pi * ((Rtank + tMLI) ** 2) + 2 * np.pi * (Rtank + tMLI) * ls

mMLI = rhoMLI * tMLI * (Aout - Atank)
print(f'Surface Area: {Atank} [m^2]')
print(f'Nbar:{Nbar} [layer/cm]')
print(f'Number of Layers: {N} [layers]')
print(f'Allowable Heat Flux: {qtank} [W / m^2]')
print(f'MLI Heat Flux: {qMLI} [W / m^2]')
print(f'Qleak: {Qleak} [W]')
print(f'MLI Heat Ingress Rate: {qMLI * Atank} [W]')
print(f'MLI Thickness: {tMLI} [m]')
print(f'MLI Mass: {mMLI} [kg]')