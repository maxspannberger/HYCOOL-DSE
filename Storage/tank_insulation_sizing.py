"""
Size the MLI insulation layer of the LH2 tank.
"""
import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq
from tank_size_shape import Geometry
# Filling conditions
Pfill = ...
Pvent = ...
rhoLH2 = ...
Ef = PropsSI('U', 'P', Pfill , 'D', rhoLH2, 'parahydrogen')
Ei = PropsSI('U', 'P', Pvent , 'D', rhoLH2, 'parahydrogen')
tauH = 24 * 3600

# Tank geometry
ls = Geometry.ls
twall = ...
Rtank = Geometry.a
Rtank_in = Rtank = twall

SF = 1.5
Qleak = (Ef - Ei) / tauH
Qtank = SF * Qleak

# Temperature
Ts = 298  # [K]
Tc = 20.3  # [K]
Tm = 298  # [K]

# Lockheed MLI Parameters
P = 10e-4 * 133.322  # [Pa]
Cs = 2.4e-4 * (0.017 + 7e-6 * (800 - Tm) + 0.0228 * np.log(Tm))
Cg = 1.46e4
Cr = 4.944e-10
emissivity =  0.043



# Calculate Nbar
Nbar = (( (Cr * emissivity * ((Ts ** 4.67) - (Tc ** 4.67)) + Cg * P * ((Ts ** 0.52) - (Tc ** 0.52))) ) / (0.815 * Cs * ((Ts ** 2) - (Tc ** 2)))) ** (1 / 2.63)

# Calculate qleak and Aout as functions of the number of MLI layers N
def qleak_func(N):
    qs = (Cs * (Nbar ** 2.63) * ((Ts ** 2) - (Tc ** 2))) / (2 * (N + 1))
    qr = (Cr * emissivity * ((Ts ** 4.67) - (Tc ** 4.67))) / N
    qg = Cg * P * ((Ts ** 0.52) - (Tc ** 0.52)) / N
    return qs + qr + qg


def Aout_func(N):
    return 2 * np.pi * (Rtank_in + N / Nbar) * (2 * (Rtank_in + N / Nbar) + ls)


def residual(N):
    return Qtank - qleak_func(N) * Aout_func(N)


# Solve Qtank = qleak * Aout for the number of MLI layers N
N = brentq(residual, 1e-6, 1e4)

qleak = qleak_func(N)
Aout = Aout_func(N)

tMLI = N / Nbar
