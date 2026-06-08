from scipy.optimize import brentq
from pathlib import Path
import numpy as np
import properties
from CoolProp.CoolProp import PropsSI

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

#import calculated tank dimensions
from geomDesign import GeomDesign
gd = GeomDesign(p_vent=2.0, p_fill=1.2, y_max=0.97)
geom = gd.calculateTankGeometry(V_tank=12.3, phi=1.0, psi=1.0, Lambda=0.5)

# hard coded dimensions for testing
D = 2.5 # m, inner diameter of the tank
L = 2.8 # m, length of the tank
# for Sphericaltank:
V = 7.3808 # m^3, volume of the tank
d = (6*V/np.pi)**(1/3) # m, diameter of the spherical tank with the same volume as the cylindrical tank

D = geom.a * 2  # m, inner diameter of the tank (from geomDesign)
L = geom.ls     # m, length of the cylindrical section of the tank (from geomDesign)

spherical = False # boolean, whether the tank is spherical or cylindrical
f_ullage = 0.0425 # fraction of tank volume reserved for ullage (empty space to allow for expansion of the liquid)

tank_options = {
    'LH2': {'P_int': 300000, 'P_ext': 37600}, # internap pressure of 3 bar, external 376 hPa (atmospheric pressure at FL250)
    'cCH2': {'P_int': 3.5e7, 'P_ext': 37600}, # internal pressure of 350 bar, external 376 hPa (atmospheric pressure at FL250)
    'sLH2': {'P_int': 30000, 'P_ext': 101325}, # internal pressure of 0.3 bar, external pressure of 1 bar (atmospheric pressure on ground)
    'sLH2_p':  {'P_int': 500000,  'P_ext': 37600,  'T': 20.0},  # 5 bar, 20 K
}

material_options = {
    'Al-2219-T87': {'E': 73.1e9, 'nu': 0.33, 'S': 300e6, 'S_t': 476e6, 'density': 2840},
}

def t_hoop_stress(P_int, P_ext, D, S_t, spherical=False):
    "calculate tank wall thickness based on hoop stress"
    S = S_t / 2 # apply safety factor to ultimate tensile strength = 2
    if spherical:
        t = (D*(P_int - P_ext))/(4*S)
    else:
        t = (D*(P_int - P_ext))/(2*S) # if tank is cylindrical, the hoop stress is 1/2 the longitudinal stress, so we use 2S instead of 4S
    return t

def t_windenburg_trilling(p_diff: float, D: float, L: float,
                          E: float, nu: float,
                          SF: float = 2.0, # adjust safety factor if needed
                          t_min: float = 0.001) -> float:
    knockdown_factor = 0.75**2  # constant for cylinder
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

def t_spherical_buckling(p_diff, d, E, nu, SF=2.0, t_min=1e-4):
    p_design = p_diff * SF
    converged = False
    max_iter = 100
    i = 0
    t  = 0.0
    while not converged and i < max_iter:
        t_new = (d/2) * np.sqrt((p_design*np.sqrt(3*(1 - nu**2)))/(2*E))
        # NASA SP-8032 knockdown factor — Figure 2, Equation 3
        lam = (12*(1 - nu**2))**0.25 * (R/t_new)**0.5 * 2*np.sin(phi/2)
        knockdown_factor = 0.14 + 3.2 / lam**2

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
t_sLH2 = t_windenburg_trilling(p_diff, d, L, mat['E'], mat['nu'], SF=3.0, t_min=1e-3)
# spherical tanks
t_LH2_s = t_hoop_stress(tank_options['LH2']['P_int'], tank_options['LH2']['P_ext'], D, mat['S_t'], spherical=True)
t_cCH2_s = t_hoop_stress(tank_options['cCH2']['P_int'], tank_options['cCH2']['P_ext'], D, mat['S_t'], spherical=True)
t_sLH2_s = t_spherical_buckling(p_diff, d, mat['E'], mat['nu'], SF=3.0, t_min=1e-3)
# sub-cooled at internal pressure
t_sLH2_p = t_hoop_stress(
    tank_options['sLH2_p']['P_int'],
    tank_options['sLH2_p']['P_ext'],
    D, mat['S_t'], spherical=False)

# calculate mass of the tank wall for each case
def calculate_tank_mass_c(t, D, L, density):
    SA = (np.pi * D * L) + (4 * np.pi * (D/2)**2)  # surface area of the cylindrical tank
    m_wall = SA * t * density
    return m_wall

def calculate_tank_mass_s(t, d, density):
    SA = 4 * np.pi * (d/2)**2  # surface area of the spherical tank
    m_wall = SA * t * density
    return m_wall

def inner_volume_c(D, L):
    return (np.pi * (D/2)**2 * L) + ((4/3) * np.pi * (D/2)**3)  # volume of cylinder + volume of two hemispherical end caps

def inner_volume_s(d):
    return (4/3) * np.pi * (d/2)**3  # volume of sphere

def calculate_LH2_mass_c(D, L, rho_h2):
    V = inner_volume_c(D, L)
    m_LH2 = V * rho_h2 * (1 - f_ullage)  # account for ullage
    return m_LH2

def calculate_LH2_mass_s(d, rho_h2):
    V = inner_volume_s(d)
    m_LH2 = V * rho_h2 * (1 - f_ullage)  # account for ullage
    return m_LH2

def gravimetric_eff(m_LH2, m_wall):
    return m_LH2 / (m_LH2 + m_wall)


def liquid_density_at_state(T, P, fluid='parahydrogen'):
    """Return liquid density for a T,P state or NaN if the state is not liquid."""
    try:
        p_sat = PropsSI('P', 'T', T, 'Q', 0, fluid)
        if P < p_sat:
            return np.nan
        return PropsSI('D', 'T', T, 'P', P, fluid)
    except ValueError:
        return np.nan


def gravimetric_efficiency_at_state(T, P, P_ext, fluid='parahydrogen'):
    """Evaluate cylindrical-tank gravimetric efficiency at a liquid T,P state."""
    rho = liquid_density_at_state(T, P, fluid)
    if not np.isfinite(rho):
        return np.nan

    t_wall = t_hoop_stress(P, P_ext, D, mat['S_t'], spherical=False)
    if t_wall <= 0:
        return np.nan

    m_wall = calculate_tank_mass_c(t_wall, D, L, mat['density'])
    m_h2 = calculate_LH2_mass_c(D, L, rho)
    return gravimetric_eff(m_h2, m_wall)


def grid_search_gravimetric_efficiency(T_values, P_values, P_ext, label):
    """Grid-search T/P to maximize gravimetric efficiency for a given external pressure."""
    best = {
        'label': label,
        'eff': -np.inf,
        'T': np.nan,
        'P': np.nan,
    }

    for T in T_values:
        for P in P_values:
            eff = gravimetric_efficiency_at_state(T, P, P_ext)
            if np.isfinite(eff) and eff > best['eff']:
                best.update({'eff': eff, 'T': T, 'P': P})

    return best


def build_efficiency_grid(T_values, P_values, P_ext):
    """Evaluate gravimetric efficiency on a T/P grid."""
    eff_grid = np.full((len(T_values), len(P_values)), np.nan)
    best = {
        'eff': -np.inf,
        'T': np.nan,
        'P': np.nan,
    }

    for i, T in enumerate(T_values):
        for j, P in enumerate(P_values):
            eff = gravimetric_efficiency_at_state(T, P, P_ext)
            eff_grid[i, j] = eff
            if np.isfinite(eff) and eff > best['eff']:
                best.update({'eff': eff, 'T': T, 'P': P})

    return eff_grid, best


def plot_efficiency_grid(T_values, P_values, eff_grid, best, title, output_path):
    """Save a contour heatmap of gravimetric efficiency over the grid."""
    if plt is None:
        print(f"matplotlib not available; skipping plot for {title}")
        return

    P_bar = np.asarray(P_values) / 1e5
    T_vals = np.asarray(T_values)
    P_mesh, T_mesh = np.meshgrid(P_bar, T_vals)
    masked = np.ma.masked_invalid(eff_grid)

    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    levels = 24
    contour = ax.contourf(P_mesh, T_mesh, masked, levels=levels, cmap='viridis')
    ax.contour(P_mesh, T_mesh, masked, levels=levels, colors='k', alpha=0.12, linewidths=0.5)
    fig.colorbar(contour, ax=ax, label='Gravimetric efficiency')

    if np.isfinite(best['eff']):
        ax.scatter(best['P'] / 1e5, best['T'], s=70, c='red', edgecolors='white', linewidths=1.0, zorder=5)
        ax.annotate(
            f"best: {best['eff']:.4f}\n{best['T']:.2f} K, {best['P']/1e5:.2f} bar",
            (best['P'] / 1e5, best['T']),
            textcoords='offset points',
            xytext=(10, 10),
            fontsize=9,
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.55),
        )

    ax.set_title(title)
    ax.set_xlabel('Pressure [bar]')
    ax.set_ylabel('Temperature [K]')
    ax.grid(True, alpha=0.2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

#different densities for LH2 at different conditions
rho_LH2 = PropsSI('D', 'P', 1 * 100000, 'T', 20, 'parahydrogen')  # kg/m^3
rho_cCH2 = PropsSI('D', 'P', 35 * 100000, 'T', 35, 'parahydrogen')  # kg/m^3
rho_sLH2 = PropsSI('D', 'P', 0.3 * 100000, 'T', 16, 'parahydrogen')  # kg/m^3
rho_sLH2_p = PropsSI('D', 'T', 20.0, 'P', 5e5, 'parahydrogen')

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
    print('sLH2_p: t =', t_sLH2_p, ", m_wall =", calculate_tank_mass_c(t_sLH2_p, D, L, mat['density']), ", m_H2 =", calculate_LH2_mass_c(D, L, rho_sLH2_p), ", eff =", gravimetric_eff(calculate_LH2_mass_c(D, L, rho_sLH2_p), calculate_tank_mass_c(t_sLH2_p, D, L, mat['density'])))  
    print('Densities (LH2, cCH2, sLH2):', rho_LH2, rho_cCH2, rho_sLH2)

    # Grid search over liquid states for the two cases that are pressure/temperature sensitive.
    lh2_T = np.linspace(19.5, 22.0, 60)
    lh2_P = np.linspace(1.0e5, 5.0e5, 80)
    slh2p_T = np.linspace(15, 20.0, 60)
    slh2p_P = np.linspace(1.0e5, 8.0e5, 80)

    best_LH2 = grid_search_gravimetric_efficiency(lh2_T, lh2_P, tank_options['LH2']['P_ext'], 'LH2')
    best_sLH2_p = grid_search_gravimetric_efficiency(slh2p_T, slh2p_P, tank_options['sLH2_p']['P_ext'], 'sLH2_p')

    lh2_grid, lh2_grid_best = build_efficiency_grid(lh2_T, lh2_P, tank_options['LH2']['P_ext'])
    slh2p_grid, slh2p_grid_best = build_efficiency_grid(slh2p_T, slh2p_P, tank_options['sLH2_p']['P_ext'])

    print('\nGrid search bests:')
    print(f"  {best_LH2['label']}: eff={best_LH2['eff']:.6f} at T={best_LH2['T']:.3f} K, P={best_LH2['P'] / 1e5:.3f} bar")
    print(f"  {best_sLH2_p['label']}: eff={best_sLH2_p['eff']:.6f} at T={best_sLH2_p['T']:.3f} K, P={best_sLH2_p['P'] / 1e5:.3f} bar")

    plot_efficiency_grid(
        lh2_T,
        lh2_P,
        lh2_grid,
        lh2_grid_best,
        'LH2 gravimetric efficiency grid search',
        Path('outputs') / 'lh2_grid_search_heatmap.png',
    )
    plot_efficiency_grid(
        slh2p_T,
        slh2p_P,
        slh2p_grid,
        slh2p_grid_best,
        'sLH2-p gravimetric efficiency grid search',
        Path('outputs') / 'slh2p_grid_search_heatmap.png',
    )