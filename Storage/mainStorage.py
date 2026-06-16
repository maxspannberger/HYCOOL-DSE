"""
Sizes the tank volume, inner lining, baffles, insulation, and outer wall.
"""

# Import statements
from tank_dimensions_sizing import dimensionSizing
from tank_insulation_sizing import insulationSizing
from tank_outer_wall_sizing import outerSizing
from tank_baffle_sizing import baffleSizing
from CoolProp.CoolProp import PropsSI

# =============================================================================
# DESIGN INPUTS
# =============================================================================

# Mission / loading
BAR    = 1e5          # [Pa]  bar-to-Pascal conversion
mLH2   = 610          # [kg]  mass of LH2
pfill  = 1.0  * BAR  # [Pa]  fill pressure
pvent  = 1.75 * BAR  # [Pa]  venting pressure
pext_SLS   = 101325  # [Pa]  ambient pressure at SLS
pext_FL250 =  37600  # [Pa]  ambient pressure at FL250
beta   = 0.55         # [-]   crashed diameter coefficient [Castro et al.]

# Tank geometry
Lambda = 0.467         # [-]   cylinder-to-total-length ratio

# LH2 thermodynamic state
T_fill  = 15          # [K]       LH2 temperature at fill
T_vent  = 20.3        # [K]       LH2 temperature at vent
tau_H   = 24          # [h]       hold time

# MLI insulation
Ts     = 298          # [K]       warm-side (ambient) temperature
Tc     = 20.3         # [K]       cold-side (LH2) temperature
Pvac   = 1e-5         # [torr]    MLI interstitial pressure
rhoMLI = 20           # [kg/m^3]  MLI bulk density

# Baffles
s_baffle = 0.5        # [m]   baffle spacing

# Material properties — Al-2219-T87 inner liner
Al2219T87 = {'E': 85.46e9, 'nu': 0.3184, 'S': 526e6, 'S_t': 717e6, 'density': 2825}

# Material properties — Teijin ITS50 CFRP outer wall
E_ITS50       = {'E1': 168.0, 'E2': 9.0,  'E3': 9.0}
nu_ITS50      = {'nu12': 0.1,  'nu13': 0.1, 'nu23': 0.3}
G_ITS50       = {'G12': 5.0,   'G13': 5.0,  'G23': 3.7}
rhoCFRP_ITS50 = {'fibre': 1800, 'resin': 1195}
angles        = [0, 45, -45, 90, 90, -45, 45, 0]

# =============================================================================
# TANK SIZING
# =============================================================================

# 1. Tank dimensions and inner liner
d = dimensionSizing(pfill, pvent)
yl_0, ullage_factor = d.calculateInitialLiquidMassFraction()
rho_H2_fill = PropsSI('D', 'P', pfill, 'T', T_fill, 'parahydrogen')
rho_H2_vent = PropsSI('D', 'P', pvent, 'T', T_vent, 'parahydrogen')
V_tank = d.calculateTankVolume(rho_H2=rho_H2_vent, m_H2=mLH2, yl_0=yl_0)
dimensions = d.calculateTankGeometry(Al2219T87, pfill, pext_FL250, V_tank, ullage_factor, Lambda=Lambda)
dimensions.print_summary()

# 2. Baffles
baffleData = baffleSizing(Al2219T87, mLH2).size(R=dimensions.R, ls=dimensions.ls, s=s_baffle, rho_LH2=rho_H2_vent)
baffleData.print_summary()

# 3. MLI insulation
Ef    = PropsSI('U', 'P', pfill, 'D', rho_H2_fill, 'parahydrogen')
Ei    = PropsSI('U', 'P', pvent, 'D', rho_H2_fill, 'parahydrogen')
Qleak = mLH2 * (Ei - Ef) / (tau_H * 3600)
ins   = insulationSizing(Ts=Ts, Tc=Tc, P=Pvac, rhoMLI=rhoMLI)
insulationData = ins.size(Qleak=Qleak, Atank=dimensions.A_in, Rtank=dimensions.Rin, ls=dimensions.ls)
insulationData.print_summary()

# 4. Outer CFRP wall
m_partial = dimensions.m_in + baffleData.mBaffle + insulationData.mMLI
outerData = outerSizing(E_ITS50, nu_ITS50, G_ITS50, rhoCFRP_ITS50, angles).size(
    Rin=dimensions.Rin,
    tMLI=insulationData.tMLI,
    twall=dimensions.t_in,
    ls=dimensions.ls,
    m_empty=m_partial,
)
outerData.print_summary()

# =============================================================================
# FINAL TANK MASSES
# =============================================================================
mtank_empty      = dimensions.m_in + baffleData.mBaffle + insulationData.mMLI + outerData.mCFRP
mtank_full       = mtank_empty + mLH2
t_total          = dimensions.t_in + insulationData.tMLI + outerData.tout
gravimetric_index = mLH2 / mtank_full

RESET  = '\033[0m'
CYAN   = '\033[96m'
ORANGE = '\033[38;5;208m'
BOLD   = '\033[1m'

title  = 'FINAL TANK MASSES'
width  = 40
border = '+' + '-' * (width - 2) + '+'
print()
print(f"{BOLD}{border}{RESET}")
print(f"{BOLD}|{CYAN}{title:^{width-2}}{RESET}{BOLD}|{RESET}")
print(f"{BOLD}{border}{RESET}")
print(f"{BOLD}|{RESET}  {'Empty tank mass':<20}{ORANGE}{mtank_empty:>10.2f} kg{RESET}  {BOLD}|{RESET}")
print(f"{BOLD}|{RESET}  {'Full tank mass':<20}{ORANGE}{mtank_full:>10.2f} kg{RESET}  {BOLD}|{RESET}")
print(f"{BOLD}|{RESET}  {'Total wall thickness':<20}{ORANGE}{t_total*1e3:>10.2f} mm{RESET}  {BOLD}|{RESET}")
print(f"{BOLD}|{RESET}  {'Gravimetric index':<20}{ORANGE}{gravimetric_index:>10.4f} -{RESET}  {BOLD}|{RESET}")
print(f"{BOLD}{border}{RESET}")
