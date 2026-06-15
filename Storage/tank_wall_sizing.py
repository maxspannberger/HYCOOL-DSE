"""
Sizes the inner aluminium lining of the tank and the outer CFRP wall. 
"""

from CoolProp.CoolProp import PropsSI
import numpy as np
from dataclasses import dataclass
import properties as props
from tank_insulation_sizing import Geometry

# Material Properties of Al-2219-T87
Al2219T87 = {'E': 85.46e9, 'nu': 0.3184, 'S': 526e6, 'S_t': 717e6, 'density': 2825}
g = 9.81 # [m/s^2]
R = 0.8913  # [m]
ls = 2.1789  # [m]
twall = 0.002  # [m]
tMLI = 0.008218  # [m]
m_wall = twall * (4 * np.pi * (R ** 2) + 2 * ls * R * np.pi) * Al2219T87['density']
m_MLI = 3.695318624751127  # [kg]
m_baffles = 19.7134626417  # [kg]
m_LH2 = 600  # [kg]

print(f'Aluminium Liner Mass: {m_wall} [kg]')
# Material Properties of Teijin ITS50
E = {
    'E1': 168.0,
     'E2': 9.0,
     'E3': 9.0
}

nu = {
    'nu12': 0.1,
    'nu13': 0.1,
    'nu23': 0.3,
}

G = {
    'G12': 5,
    'G13': 5,
    'G23': 3.7
}

rhoCFRP = {
    'fibre': 1800,
    'resin': 1195
}

angles = [0, 45, -45, 90, 90, 45, -45, 0]

n = len(angles)

f_fibre = 0.65
f_matrix = 0.35

# Extract Useful Material Properties
rhoBulk = 0.65*rhoCFRP['fibre'] + 0.35*rhoCFRP['resin']
E1 = E['E1']
E2 = E['E2']
G12 = G['G12']
nu12 = nu['nu12']
nu21 = nu['nu12'] * (E2/E1)
R = 0.8 #... Geometry.a

# Calculate Reduced Stiffness Matrix Q
den = 1 - (nu12 * nu21)
Q11 = E1 / den
Q22 = E2 / den
Q12 = (nu12 * E2) / den
Q66 = G12
Qred  = np.array(([Q11, Q12, 0],
                 [Q12, Q22, 0],
                 [0, 0, Q66]))

# Check Values
print(f'Q11: {Q11}')
print(f'Q22: {Q22}')
print(f'Q12: {Q12}')
print(f'Q66: {Q66}')
print(Qred)

# Calculate the Transformed Stiffness Matrix
def Qbar(theta, Qred):
    theta_rad = np.radians(theta)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    c2 = c ** 2; s2 = s**2
    cs = c * s

    T1 = np.array([
        [c2, s2, 2 * cs],
        [s2, c2, -2 * cs],
        [-cs, cs, c2 - s2]
    ])

    T2 = np.array([
        [c2, s2, cs],
        [s2, c2, -cs],
        [-2 * cs, 2 * cs, c2 - s2]
    ])

    T1_inv = np.linalg.inv(T1)
    Qbar = T1_inv @ Qred @ T2
    
    return Qbar

# Calculate the Qbar matrices for each ply

Qbar_0 = Qbar(0, Qred)
Qbar_plus45 = Qbar(11.52, Qred)
Qbar_min45 = Qbar(-11.52, Qred)
Qbar_90 = Qbar(90, Qred)

print(f'\nQbar at 0 deg: {Qbar_0}\n')
print(f'Qbar at +45 deg: {Qbar_plus45}\n')
print(f'Qbar at -45 deg: {Qbar_min45}\n')
print(f'Qbar at 90 deg: {Qbar_90}\n')

tCFRP = 0.000215
h = n * tCFRP

# Calculate the A matrix
A = np.zeros((3, 3))
for angle in angles:
    A += Qbar(angle, Qred) * tCFRP

print(f'\n A Matrix: {A}')

a = np.linalg.inv(A)
print(f'\n a Matrix: {a}')

# Calculate the D matrix
D = np.zeros((3, 3))
z = np.linspace(-h/2, h/2, n + 1)

for k, angle in enumerate(angles):
    zk = z[k + 1]
    zk_1 = z[k]
    D += (1/3) * Qbar(angle, Qred) * (zk**3 - zk_1**3)

d = np.linalg.inv(D)

# Calcuclate the quasi-isotropic feedback
Ex = 1 / (h * a[0, 0])
Ey = 1 / (h * a[1, 1])
nuxy = - a[0, 1] / a[0, 0]
Gxy = 1 / (h * a[2, 2])

print(f'Ex: {Ex}')
print(f'Ey: {Ey}')
print(f'nuxy: {nuxy}')
print(f'Gxy: {Gxy}')

# Calculation of critical axial buckling stress

sigma_cr_Al = Al2219T87['E'] * (twall / (R + tMLI + 2 * twall)) / np.sqrt(3 * (1 - (Al2219T87['nu'] ** 2)))

# Calculation of the axial buckling load
Rout = R + twall + tMLI + h
sigma_cr_CFRP = (Ex * (tCFRP / Rout) / np.sqrt(3 * (1 - (nuxy ** 2)))  * 0.2) * 10e9
Aout = 4 * np.pi * (Rout ** 2) + 2 * np.pi * Rout * ls 
Ain =  4 * np.pi * (R ** 2) + 2 * np.pi * R * ls 

print(f'CFRP Critical Buckling Load: {sigma_cr_CFRP} [Pa]')
print(f'Aluminium Critical Load : {sigma_cr_Al}[Pa]')

m_CFRP = h * rhoBulk * Aout 
m_al = 0.002 * Al2219T87['density'] * Ain
m_tank = m_LH2 + m_MLI + m_wall + m_baffles + m_CFRP
print(f'CFRP Mass: {m_CFRP} [kg]')
print(f'Aluminium Mass: {m_al} [kg]')
print(f'Tank Mass: {m_tank} [kg]')

sigma_x = (9 * g * m_tank) / (2 * np.pi * ((Rout ** 2) - (R ** 2)))
print(f'Compression Load: {sigma_x} [Pa]')
print(f'Thickness: {h} [m]')

if sigma_x < sigma_cr_CFRP:
    print(f'WE HAVE A TANK') 
else:
    print(f"WOMP WOMP")