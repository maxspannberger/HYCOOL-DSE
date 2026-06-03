from scipy.optimize import brentq
import numpy as np
import properties
from CoolProp.CoolProp import PropsSI

#import calculated tank dimensions
from geomDesign import GeomDesign
gd = GeomDesign(p_vent=2.0, p_fill=1.2, y_max=0.97)
geom = gd.calculateTankGeometry(V_tank=12.3, phi=1.0, psi=1.0, Lambda=0.5)

# hard coded dimensions for testing
D = 2.5 # m, inner diameter of the tank
L = 2.8 # m, length of the tank

D = geom.a * 2  # m, inner diameter of the tank (from geomDesign)
L = geom.ls     # m, length of the cylindrical section of the tank (from geomDesign)

spherical = False # boolean, whether the tank is spherical or cylindrical
f_ullage = 0.1 # fraction of tank volume reserved for ullage (empty space to allow for expansion of the liquid)

tank_options = {
    'LH2': {'P_int': 300000, 'P_ext': 37600}, # internap pressure of 3 bar, external 376 hPa (atmospheric pressure at FL250)
    'cCH2': {'P_int': 3.5e7, 'P_ext': 37600}, # internal pressure of 350 bar, external 376 hPa (atmospheric pressure at FL250)
    'sLH2': {'P_int': 30000, 'P_ext': 101325}, # internal pressure of 0.3 bar, external pressure of 1 bar (atmospheric pressure on ground)
}

material_options = {
    'Al-2219-T87': {'E': 73.1e9, 'nu': 0.33, 'S': 300e6, 'S_t': 476e6, 'density': 2840}, # E in Pa, S in Pa, density in kg/m^3
}

def t_hoop_stress(P_int, P_ext, D, S_t, spherical=False):
    "calculate tank wall thickness based on hoop stress"
    S = S_t / 2 # apply safety factor to tensile strength = 2
    if spherical:
        t = (D*(P_int - P_ext))/(4*S)
    else:
        t = (D*(P_int - P_ext))/(2*S) # if tank is cylindrical, the hoop stress is 1/2 the longitudinal stress, so we use 2S instead of 4S
    return t

def t_windenburg_trilling(p_diff: float, D: float, L: float,
                          E: float, nu: float,
                          SF: float = 3.0, # adjust safety factor if needed
                          t_min: float = 0.001) -> float:
    knockdown_factor = 0.75  # constant for cylinder
    p_design = p_diff * SF / knockdown_factor   # design pressure including safety factor

    def residual(t):
        x   = t / D
        lam = L / D
        denom = lam - 0.45 * x**0.5
        if denom <= 0:
            return np.inf
        p_cr = (2.42 * E / (1 - nu**2)**0.75) * (x**2.5 / denom)
        return p_cr - p_design

    # Bracket: t_lo well below solution, t_hi well above
    t_lo = 1e-4
    t_hi = D / 2.0   # physically cannot exceed radius

    try:
        t_sol = brentq(residual, t_lo, t_hi, xtol=1e-7)
    except ValueError:
        # If no root found, fall back to infinite-cylinder formula
        t_sol = D * ((p_design * (1 - nu**2) / (2 * E)) ** (1/3))
        print("Fell back to infinite-cylinder formula, solution may be inaccurate.")

    return max(t_min, t_sol)

def t_spherical_buckling(p_diff, D, E, nu, SF=3.0, t_min=1e-4):
    p_design = p_cr = p_diff * SF
    converged = False
    max_iter = 100
    i = 0
    t  = 0.0
    while not converged and i < max_iter:
        t_new = D * np.sqrt((p_design * np.sqrt(3*(1 - nu**2)))/(8*E))
        knockdown_factor = 0.124*(D/(2*t_new))**(-0.6)
        p_design = p_diff * SF / knockdown_factor

        if i > 0 and abs(t_new - t) / t < 1e-6:
            converged = True
        t = t_new
        i += 1
    return max(t_min, t)

# For cases where internal pressure is greater than external, use hoop-stress sizing.
# If external pressure exceeds internal (vacuum or sub-atmospheric internal),
# the failure mode is buckling — use the Windenburg-Trilling external-pressure formula.
mat = material_options['Al-2219-T87']
t_LH2 = t_hoop_stress(tank_options['LH2']['P_int'], tank_options['LH2']['P_ext'], D, mat['S_t'], spherical)
t_cCH2 = t_hoop_stress(tank_options['cCH2']['P_int'], tank_options['cCH2']['P_ext'], D, mat['S_t'], spherical)
# sLH2 has lower internal pressure than external (external > internal),
p_diff = tank_options['sLH2']['P_ext'] - tank_options['sLH2']['P_int']
t_sLH2 = t_windenburg_trilling(p_diff, D, L, mat['E'], mat['nu'], SF=3.0, t_min=1e-3)
# spherical tanks
t_LH2_s = t_hoop_stress(tank_options['LH2']['P_int'], tank_options['LH2']['P_ext'], D, mat['S_t'], spherical=True)
t_cCH2_s = t_hoop_stress(tank_options['cCH2']['P_int'], tank_options['cCH2']['P_ext'], D, mat['S_t'], spherical=True)
t_sLH2_s = t_spherical_buckling(p_diff, D, mat['E'], mat['nu'], SF=3.0, t_min=1e-3)

# calculate mass of the tank wall for each case
def calculate_tank_mass_c(t, D, L, density):
    SA = (np.pi * D * L) + (4 * np.pi * (D/2)**2)  # surface area of the cylindrical tank
    m_wall = SA * t * density
    return m_wall

def calculate_tank_mass_s(t, D, density):
    SA = 4 * np.pi * (D/2)**2  # surface area of the spherical tank
    m_wall = SA * t * density
    return m_wall

def inner_volume_c(D, L):
    return (np.pi * (D/2)**2 * L) + ((4/3) * np.pi * (D/2)**3)  # volume of cylinder + volume of two hemispherical end caps

def inner_volume_s(D):
    return (4/3) * np.pi * (D/2)**3  # volume of sphere

def calculate_LH2_mass_c(D, L, rho_h2):
    V = inner_volume_c(D, L)
    m_LH2 = V * rho_h2 * (1 - f_ullage)  # account for ullage
    return m_LH2

def calculate_LH2_mass_s(D, rho_h2):
    V = inner_volume_s(D)
    m_LH2 = V * rho_h2 * (1 - f_ullage)  # account for ullage
    return m_LH2

def gravimetric_eff(m_LH2, m_wall):
    return m_LH2 / (m_LH2 + m_wall)

#different densities for LH2 at different conditions
rho_LH2 = PropsSI('D', 'P', 1 * 100000, 'T', 20, 'parahydrogen')  # kg/m^3
rho_cCH2 = PropsSI('D', 'P', 350 * 100000, 'T', 35, 'parahydrogen')  # kg/m^3
rho_sLH2 = PropsSI('D', 'P', 0.3 * 100000, 'T', 16, 'parahydrogen')  # kg/m^3

# Calculate H2 masses and gravimetric efficiency for each case
m_wall_LH2 = calculate_tank_mass_c(t_LH2, D, L, mat['density'])
m_LH2 = calculate_LH2_mass_c(D, L, rho_LH2)
eff_LH2 = gravimetric_eff(m_LH2, m_wall_LH2)

m_wall_cCH2 = calculate_tank_mass_c(t_cCH2, D, L, mat['density'])
m_cCH2 = calculate_LH2_mass_c(D, L, rho_cCH2)
eff_cCH2 = gravimetric_eff(m_cCH2, m_wall_cCH2)

m_wall_sLH2 = calculate_tank_mass_c(t_sLH2, D, L, mat['density'])
m_sLH2 = calculate_LH2_mass_c(D, L, rho_sLH2)
eff_sLH2 = gravimetric_eff(m_sLH2, m_wall_sLH2)

m_wall_LH2_s = calculate_tank_mass_s(t_LH2_s, D, mat['density'])
m_LH2_s = calculate_LH2_mass_s(D, rho_LH2)  # density of LH2 at 20K in kg/m^3
eff_LH2_s = gravimetric_eff(m_LH2_s, m_wall_LH2_s)

m_wall_cCH2_s = calculate_tank_mass_s(t_cCH2_s, D, mat['density'])
m_cCH2_s = calculate_LH2_mass_s(D, rho_cCH2)  # density of cCH2 at 100 bar in kg/m^3
eff_cCH2_s = gravimetric_eff(m_cCH2_s, m_wall_cCH2_s)

m_wall_sLH2_s = calculate_tank_mass_s(t_sLH2_s, D, mat['density'])
m_sLH2_s = calculate_LH2_mass_s(D, rho_sLH2)
eff_sLH2_s = gravimetric_eff(m_sLH2_s, m_wall_sLH2_s)

if __name__ == "__main__":
    print('LH2: t =', t_LH2, ", m_wall =", m_wall_LH2, ", m_H2 =", m_LH2, ", eff =", eff_LH2)
    print('cCH2: t =', t_cCH2, ", m_wall =", m_wall_cCH2, ", m_H2 =", m_cCH2, ", eff =", eff_cCH2)
    print('sLH2: t =', t_sLH2, ", m_wall =", m_wall_sLH2, ", m_H2 =", m_sLH2, ", eff =", eff_sLH2)
    print('LH2 (spherical): t =', t_LH2_s, ", m_wall =", m_wall_LH2_s, ", m_H2 =", m_LH2_s, ", eff =", eff_LH2_s)
    print('cCH2 (spherical): t =', t_cCH2_s, ", m_wall =", m_wall_cCH2_s, ", m_H2 =", m_cCH2_s, ", eff =", eff_cCH2_s)
    print('sLH2 (spherical): t =', t_sLH2_s, ", m_wall =", m_wall_sLH2_s, ", m_H2 =", m_sLH2_s, ", eff =", eff_sLH2_s)
    print('Densities (LH2, cCH2, sLH2):', rho_LH2, rho_cCH2, rho_sLH2)