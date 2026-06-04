from TurbineSizing import GasTurbineCycle
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich.columns import Columns
from rich import box
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj_w_units import CEA_Obj
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


engine = GasTurbineCycle()
engine.size()
engine.report()

cea = CEA_Obj(
    oxName="AIR", fuelName="GH2",
    pressure_units="bar", temperature_units="K", isp_units="sec",
)

results = engine.results
design  = engine._design

_console = Console()

# ---------------- Helper functions -----------------------

def _air_gamma(p_bar, T):
    """Ratio of specific heats for air at (p_bar, T)."""
    return (PropsSI('CPMASS', 'P', p_bar*1e5, 'T', T, 'Air') /
            PropsSI('CVMASS', 'P', p_bar*1e5, 'T', T, 'Air'))

def _air_cp(p_bar, T):
    """Isobaric specific heat of air [J/kg/K] at (p_bar, T)."""
    return PropsSI('CPMASS', 'P', p_bar*1e5, 'T', T, 'Air')

def _air_rho(p_bar, T):
    """Density of air [kg/m3] via ideal gas."""
    R_air = 287.0
    return (p_bar * 1e5) / (R_air * T)

def _make_table(title, rows, color="magenta"):
    """
    Build a rich Table for one component. Returns the Table object.
    rows: list of (label, value_str, unit_str) or None for section divider.
    """
    t = Table(
        title=f"[bold cyan]{title}[/bold cyan]",
        box=box.ROUNDED,
        header_style=f"bold {color}",
        show_lines=True,
        title_justify="left",
        min_width=52,
    )
    t.add_column("Parameter", style="cyan",   justify="left",  min_width=26)
    t.add_column("Value",     style="yellow",  justify="right", min_width=10)
    t.add_column("Units",     style="dim",     justify="left",  min_width=8)
    for row in rows:
        if row is None:
            t.add_section()
        else:
            t.add_row(*row)
    return t

# ----------- Station Parameters --------------

P1, P2, Pc, P4, P5      = design["P1"], design["P2"], design["Pc"], design["P4"], design["P5"]
T1, T2, T2p, Tc, T4, T5 = design["T1"], design["T2"], design["T2p"], design["T4"], design["T4"], design["T5"]
mdot_f                   = results["mdot_f"]
mdot_air                 = results["ideal_OF"] * mdot_f
mdot_tot                 = mdot_air + mdot_f            # total gas-path mass flow through HPT

R_air                    = 287.0                        # J/kg/K

# =====================================================================
# SECTION 1: HPC SIZING  (axial, air, large mass flow, viable)
# =====================================================================

Inlet_HTR               = 0.45
Outlet_HTR              = 0.7
U_tip                   = 450.0                         # m/s
Psi                     = 0.45                          # Work coefficient, delta_h0 / U_mean
BladeChord              = 0.03                          # m
Spacing                 = 0.3                           # Inter-row gap fraction of axial chord

rho_1       = P1*1e5 / (R_air * T1)
rho_2       = P2*1e5 / (R_air * T2)
M_ax        = 0.5

g_1         = _air_gamma(P1, T1)
C_ax        = M_ax * np.sqrt(g_1 * R_air * T1)         # Axial velocity, held constant throughout HPC.

A_annulus        = mdot_air / (rho_1 * C_ax)
A_annulus_outlet = mdot_air / (rho_2 * C_ax)

inlet_tip   = np.sqrt(A_annulus        / (np.pi * (1 - Inlet_HTR**2)))
inlet_hub   = Inlet_HTR  * inlet_tip
outlet_tip  = np.sqrt(A_annulus_outlet / (np.pi * (1 - Outlet_HTR**2)))
outlet_hub  = Outlet_HTR * outlet_tip

r_mean_HPC          = 0.5 * (0.5*(inlet_tip + inlet_hub) + 0.5*(outlet_tip + outlet_hub))
r_mean_HPC_in       = 0.5*(inlet_tip + inlet_hub)
r_mean_HPC_out      = 0.5*(outlet_tip + outlet_hub)

RPM_HPC         = (U_tip / inlet_tip) * (60 / (2 * np.pi))
omega_HPC       = RPM_HPC * 2 * np.pi / 60
U_mean_HPC      = omega_HPC * r_mean_HPC

Cp_hpc          = _air_cp(0.5*(P1+P2), 0.5*(T1+T2))
Delta_h0_HPC    = Cp_hpc * (T2 - T1)
Stages_HPC      = np.ceil(Delta_h0_HPC / (Psi * U_mean_HPC**2))
L_HPC           = Stages_HPC * 2 * BladeChord * (1 + Spacing)

# =====================================================================
# SECTION 2: HPT SIZING  (axial, combustion products, large mass flow, viable)
# =====================================================================

HPT_Inlet_HTR           = 0.80
HPT_Outlet_HTR          = 0.70
U_tip_HPT               = 400.0                         # m/s
Psi_HPT                 = 1.5
BladeChord_HPT          = 0.04                          # m
Spacing_HPT             = 0.3
M_ax_HPT                = 0.3

g_4         = _air_gamma(P4, T4)
a_4         = np.sqrt(g_4 * R_air * T4)
C_ax_HPT    = M_ax_HPT * a_4

rho_4       = _air_rho(P4, T4)
rho_5       = _air_rho(P5, T5)

A_HPT_inlet  = mdot_tot / (rho_4 * C_ax_HPT)
A_HPT_outlet = mdot_tot / (rho_5 * C_ax_HPT)

HPT_inlet_tip   = np.sqrt(A_HPT_inlet  / (np.pi * (1 - HPT_Inlet_HTR**2)))
HPT_inlet_hub   = HPT_Inlet_HTR  * HPT_inlet_tip
HPT_outlet_tip  = np.sqrt(A_HPT_outlet / (np.pi * (1 - HPT_Outlet_HTR**2)))
HPT_outlet_hub  = HPT_Outlet_HTR * HPT_outlet_tip

r_mean_HPT_in  = 0.5 * (HPT_inlet_tip  + HPT_inlet_hub)
r_mean_HPT_out = 0.5 * (HPT_outlet_tip + HPT_outlet_hub)
r_mean_HPT     = 0.5 * (r_mean_HPT_in  + r_mean_HPT_out)

RPM_HPT        = (U_tip_HPT / HPT_inlet_tip) * (60 / (2 * np.pi))
omega_HPT      = RPM_HPT * 2 * np.pi / 60
U_mean_HPT     = omega_HPT * r_mean_HPT

Cp_hpt         = _air_cp(0.5*(P4+P5), 0.5*(T4+T5))
Delta_h0_HPT   = Cp_hpt * (T4 - T5)
Stages_HPT     = np.ceil(Delta_h0_HPT / (Psi_HPT * U_mean_HPT**2))
L_HPT          = Stages_HPT * 2 * BladeChord_HPT * (1 + Spacing_HPT)


# =====================================================================
# SECTION 3: Combustion Chamber (RQL, continuity-driven marching)
#
# Architecture: Rich -> Quench -> Lean
#
# Diameter at each zone is derived from continuity:
#   A = mdot / (rho * C_ax_CC)
#   D = sqrt(4*A/pi)
# rather than linearly interpolated, so the geometry is physically
# consistent with the local mass flow and temperature.
#
# The quench zone length is set by jet penetration depth rather than
# residence time: L_quench ~ 0.5 * D_quench (standard approximation).
# =====================================================================

# --- Stoichiometric O/F and adiabatic flame temperature from CEA ---
OF_range    = np.arange(1, 200, 0.5)
full_output = np.array([cea.get_Tcomb(Pc=Pc, MR=of) for of in OF_range])
OF_stoich   = OF_range[np.argmax(full_output)]          # stoichiometric O/F by mass
T_flame     = np.max(full_output)                       # adiabatic flame temperature [K]

OF_total    = results["ideal_OF"]                       # overall O/F from cycle

# --- RQL zone design parameters ---
C_ax_CC     = 40.0                                      # m/s, axial velocity through combustor
                                                        # 30-50 m/s is standard for gas turbine combustors

tau_rich    = 2.0e-3                                    # s, rich zone residence time
                                                        # H2 burns fast; 1-3 ms is typical
tau_lean    = 3.5e-3                                    # s, lean zone residence time
                                                        # longer: need to complete combustion before HPT

f_quench    = 0.20                                      # fraction of total air mass entering quench jets
                                                        # 15-25% is typical for RQL

# --- Zone air fractions ---
# f_primary: fraction of air entering rich zone, set by stoichiometric O/F
# f_secondary: remaining air added in lean zone
f_primary   = OF_stoich / OF_total                      # rich zone air fraction
f_secondary = 1.0 - f_primary - f_quench               # lean zone air fraction
                                                        # NOTE: if f_secondary < 0, f_quench is too large
                                                        # for this cycle's overall O/F. Reduce f_quench.

# --- Zone mass flows (cumulative, each zone adds air) ---
mdot_primary = mdot_f * (1.0 + OF_stoich)              # fuel + primary air only
mdot_quench  = mdot_primary + mdot_f * OF_total * f_quench  # + quench air
mdot_lean    = mdot_tot                                 # full flow in lean zone

# --- Zone mean temperatures ---
T_rich   = T_flame                                      # peak adiabatic, rich zone
T_quench = 0.5 * (T_flame + T4)                        # rough average during rapid mixing
T_lean   = T4                                           # lean zone exits at TIT

# --- Zone densities (ideal gas, combustion products approximated as air) ---
rho_rich   = Pc*1e5 / (R_air * T_rich)
rho_quench = Pc*1e5 / (R_air * T_quench)
rho_lean   = Pc*1e5 / (R_air * T_lean)

# --- Zone cross-sectional areas from continuity: A = mdot / (rho * C_ax) ---
A_rich   = mdot_primary / (rho_rich   * C_ax_CC)
A_quench = mdot_quench  / (rho_quench * C_ax_CC)
A_lean   = mdot_lean    / (rho_lean   * C_ax_CC)

# --- Zone diameters ---
D_rich   = np.sqrt(4 * A_rich   / np.pi)
D_quench = np.sqrt(4 * A_quench / np.pi)
D_lean   = np.sqrt(4 * A_lean   / np.pi)

# --- Zone lengths from residence time and volume ---
# V = tau * mdot / rho  (volume needed to achieve residence time tau)
# L = V / A
V_rich  = tau_rich * mdot_primary / rho_rich
V_lean  = tau_lean * mdot_lean    / rho_lean

L_rich   = V_rich  / A_rich
L_quench = 0.5 * D_quench                              # quench zone: jet penetration depth ~ 0.5*D
L_lean   = V_lean  / A_lean

CC_L     = L_rich + L_quench + L_lean                  # total combustor length

# --- Diameter check against HPC/HPT geometry ---
# The combustor inlet diameter should be compatible with HPC outlet mean diameter.
# The combustor exit diameter should be compatible with HPT inlet tip diameter.
D_HPC_out_mean = 2 * r_mean_HPC_out                    # mean diameter at HPC outlet
D_HPT_in_tip   = 2 * HPT_inlet_tip                     # tip diameter at HPT inlet
                                                        # NOTE: the combustor exit D_lean will likely
                                                        # differ from D_HPT_in_tip. This discrepancy
                                                        # is handled by a transition duct (diffuser/nozzle)
                                                        # between combustor and HPT -- not sized here.


# =====================================================================
# OUTPUT
# =====================================================================

_console.print()
_console.rule("[bold white]COMPONENT DIMENSIONAL SIZING SUMMARY[/bold white]")
_console.print()

tbl_HPC = _make_table("HPC  (Axial, Air)", [
    ("Stages",               f"{Stages_HPC:.0f}",              "-"),
    ("Length",               f"{L_HPC*100:.1f}",               "cm"),
    None,
    ("Inlet tip radius",     f"{inlet_tip*100:.2f}",           "cm"),
    ("Inlet hub radius",     f"{inlet_hub*100:.2f}",           "cm"),
    ("Outlet tip radius",    f"{outlet_tip*100:.2f}",          "cm"),
    ("Outlet hub radius",    f"{outlet_hub*100:.2f}",          "cm"),
    None,
    ("RPM",                  f"{RPM_HPC:.0f}",                 "rpm"),
    ("U_mean",               f"{U_mean_HPC:.1f}",              "m/s"),
    ("Specific work",        f"{Delta_h0_HPC/1e3:.1f}",        "kJ/kg"),
    None,
    ("Annulus area (in)",    f"{A_annulus:.4f}",               "m\u00b2"),
    ("Annulus area (out)",   f"{A_annulus_outlet:.4f}",        "m\u00b2"),
    ("Air density (in)",     f"{rho_1:.3f}",                   "kg/m\u00b3"),
    ("Air density (out)",    f"{rho_2:.3f}",                   "kg/m\u00b3"),
    ("mdot air",             f"{mdot_air:.3f}",                "kg/s"),
], color="blue")

tbl_HPT = _make_table("HPT  (Axial, Combustion Products)", [
    ("Stages",               f"{Stages_HPT:.0f}",              "-"),
    ("Length",               f"{L_HPT*100:.1f}",               "cm"),
    None,
    ("Inlet tip radius",     f"{HPT_inlet_tip*100:.2f}",       "cm"),
    ("Inlet hub radius",     f"{HPT_inlet_hub*100:.2f}",       "cm"),
    ("Outlet tip radius",    f"{HPT_outlet_tip*100:.2f}",      "cm"),
    ("Outlet hub radius",    f"{HPT_outlet_hub*100:.2f}",      "cm"),
    None,
    ("RPM",                  f"{RPM_HPT:.0f}",                 "rpm"),
    ("U_mean",               f"{U_mean_HPT:.1f}",              "m/s"),
    ("Specific work",        f"{Delta_h0_HPT/1e3:.1f}",        "kJ/kg"),
    None,
    ("Annulus area (in)",    f"{A_HPT_inlet:.4f}",             "m\u00b2"),
    ("Annulus area (out)",   f"{A_HPT_outlet:.4f}",            "m\u00b2"),
    ("Gas density (in)",     f"{rho_4:.3f}",                   "kg/m\u00b3"),
    ("Gas density (out)",    f"{rho_5:.4f}",                   "kg/m\u00b3"),
    ("mdot total",           f"{mdot_tot:.3f}",                "kg/s"),
], color="green")

tbl_CC = _make_table("Combustion Chamber  (RQL, H2/Air)", [
    ("Stoichiometric O/F",   f"{OF_stoich:.1f}",               "-"),
    ("Adiabatic flame T",    f"{T_flame:.0f}",                  "K"),
    ("Overall O/F (cycle)",  f"{OF_total:.1f}",                 "-"),
    None,
    ("Rich zone diameter",   f"{D_rich*100:.1f}",              "cm"),
    ("Rich zone length",     f"{L_rich*100:.1f}",              "cm"),
    ("Quench zone diameter", f"{D_quench*100:.1f}",            "cm"),
    ("Quench zone length",   f"{L_quench*100:.1f}",            "cm"),
    ("Lean zone diameter",   f"{D_lean*100:.1f}",              "cm"),
    ("Lean zone length",     f"{L_lean*100:.1f}",              "cm"),
    None,
    ("Total length",         f"{CC_L*100:.1f}",                "cm"),
    ("Max diameter (lean)",  f"{D_lean*100:.1f}",              "cm"),
    None,
    ("Axial velocity",       f"{C_ax_CC:.0f}",                 "m/s"),
    ("tau_rich",             f"{tau_rich*1e3:.1f}",            "ms"),
    ("tau_lean",             f"{tau_lean*1e3:.1f}",            "ms"),
    ("f_primary",            f"{f_primary:.3f}",               "-"),
    ("f_quench",             f"{f_quench:.3f}",                "-"),
    ("f_secondary",          f"{f_secondary:.3f}",             "-"),
    None,
    ("HPC outlet mean D",    f"{D_HPC_out_mean*100:.1f}",      "cm"),
    ("HPT inlet tip D",      f"{D_HPT_in_tip*100:.1f}",        "cm"),
], color="red")

_console.print(Columns([tbl_HPC, tbl_HPT], equal=True, expand=False))
_console.print()
_console.print(tbl_CC)
_console.print()

if f_secondary < 0:
    _console.print("[bold red]WARNING:[/bold red] f_secondary < 0: quench fraction too large for this O/F. Reduce f_quench.")


# =====================================================================
# SECTION 4: To-scale cross-section diagram
#
# Full axial cross-section, both upper and lower halves mirrored about
# the centreline (y=0). Each draw_* function accepts a `sign` argument
# (+1 for upper half, -1 for lower half) so the same call draws both.
#
# Components:
#   - HPC and HPT: annular flow region (tip to hub), mirrored
#   - Shaft: solid rectangle centred on y=0 running full engine length
#   - Combustor zones: full-diameter cylinders from -r_outer to +r_outer
#   - Transition ducts: casing and hub lines only, no fill
# =====================================================================

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_aspect('equal')
ax.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')

COL_HPC       = '#4a9eff'
COL_HPT       = '#4aff8a'
COL_CC_RICH   = "#ff3535"
COL_CC_QUENCH = "#ff9500"
COL_CC_LEAN   = "#fff235"
COL_CASING    = '#cccccc'
COL_SHAFT_FILL= '#555555'
COL_SHAFT_EDGE= '#999999'
COL_TEXT      = '#ffffff'
COL_GRID      = '#2a2a2a'
COL_DIM       = '#aaaaaa'

r_shaft = min(inlet_hub, outlet_hub, HPT_inlet_hub, HPT_outlet_hub) * 0.85

def draw_annulus(ax, x0, x1, r_tip_0, r_hub_0, r_tip_1, r_hub_1,
                 color, alpha=0.72, label=None):
    """
    Draw both halves of an annular component (turbomachinery stage).
    Upper half: y = +hub to +tip. Lower half: y = -tip to -hub.
    A separate shaft rectangle fills the hub interior.
    """
    for s in (+1, -1):
        # Annular region: outer boundary tip, inner boundary hub
        xs = [x0,          x1,          x1,          x0         ]
        ys = [s*r_tip_0,   s*r_tip_1,   s*r_hub_1,   s*r_hub_0  ]
        ax.fill(xs, ys, color=color, alpha=alpha, zorder=4)
        # Outer (tip) casing wall
        ax.plot([x0, x1], [s*r_tip_0, s*r_tip_1], color=COL_CASING, lw=1.8, zorder=5)
        # Inner (hub) wall
        ax.plot([x0, x1], [s*r_hub_0, s*r_hub_1], color=COL_CASING, lw=1.2, zorder=5)
        # Face lines
        ax.plot([x0, x0], [s*r_hub_0, s*r_tip_0], color=COL_CASING, lw=1.0, zorder=5)
        ax.plot([x1, x1], [s*r_hub_1, s*r_tip_1], color=COL_CASING, lw=1.0, zorder=5)
    if label:
        xm = 0.5*(x0 + x1)
        ym = 0.5*(0.5*(r_tip_0+r_hub_0) + 0.5*(r_tip_1+r_hub_1))
        ax.text(xm, ym, label, color=COL_TEXT, fontsize=9,
                ha='center', va='center', fontweight='bold', zorder=7)

def draw_combustor_zone(ax, x0, x1, r0, r1, color, alpha=0.60, label=None):
    """
    Draw a combustor zone spanning the full diameter (both halves).
    Filled from -r_outer to +r_outer; shaft block overlays the centre.
    """
    # Full-width fill both sides at once (one rectangle from -r to +r)
    xs = [x0,  x1,  x1,  x0]
    ys = [r0,  r1,  -r1, -r0]
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=4)
    # Outer wall upper and lower
    ax.plot([x0, x1], [ r0,  r1], color=COL_CASING, lw=2.0, zorder=5)
    ax.plot([x0, x1], [-r0, -r1], color=COL_CASING, lw=2.0, zorder=5)
    # Face lines
    ax.plot([x0, x0], [-r0,  r0], color=COL_CASING, lw=1.0, zorder=5)
    ax.plot([x1, x1], [-r1,  r1], color=COL_CASING, lw=1.0, zorder=5)
    if label:
        xm = 0.5*(x0 + x1)
        ym = 0.5*(0.5*(r0+r1))
        ax.text(xm, ym, label, color=COL_TEXT, fontsize=9,
                ha='center', va='center', fontweight='bold', zorder=7)

def draw_transition(ax, x0, x1, r_tip_0, r_hub_0, r_tip_1, r_hub_1):
    """
    Draw transition duct casing and hub lines for both halves, no fill.
    """
    for s in (+1, -1):
        ax.plot([x0, x1], [s*r_tip_0, s*r_tip_1],
                color=COL_CASING, lw=1.5, zorder=5)
        ax.plot([x0, x1], [s*r_hub_0, s*r_hub_1],
                color=COL_CASING, lw=1.0, linestyle=':', zorder=5)

def dim_arrow(ax, x1, x2, y, label):
    """Horizontal dimension arrow with label above the line."""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='<->', color=COL_DIM, lw=1.2))
    ax.text(0.5*(x1+x2), y + 0.005, label, color=COL_DIM,
            fontsize=7.5, ha='center', va='bottom', zorder=8)

# ---- Layout ----
x = 0.02
x_inlet = x

# HPC
x_HPC_start = x
x_HPC_end   = x + L_HPC
draw_annulus(ax, x_HPC_start, x_HPC_end,
             inlet_tip, inlet_hub,
             outlet_tip, outlet_hub,
             COL_HPC, label='HPC')
x = x_HPC_end

# Transition HPC -> CC
L_trans_in = 0.04
x_tr1_end  = x + L_trans_in
draw_transition(ax, x, x_tr1_end,
                outlet_tip, outlet_hub,
                D_rich/2,   r_shaft)
x = x_tr1_end

# Combustor
x_CC_start   = x
x_rich_end   = x + L_rich
x_quench_end = x_rich_end   + L_quench
x_lean_end   = x_quench_end + L_lean
draw_combustor_zone(ax, x,            x_rich_end,   D_rich/2,   D_rich/2,   COL_CC_RICH,   label='Rich')
draw_combustor_zone(ax, x_rich_end,   x_quench_end, D_rich/2,   D_quench/2, COL_CC_QUENCH, label='Quench')
draw_combustor_zone(ax, x_quench_end, x_lean_end,   D_quench/2, D_lean/2,   COL_CC_LEAN,   label='Lean')
x = x_lean_end
x_CC_end = x

# Transition CC -> HPT
L_trans_out = 0.04
x_tr2_end   = x + L_trans_out
draw_transition(ax, x, x_tr2_end,
                D_lean/2,      r_shaft,
                HPT_inlet_tip, HPT_inlet_hub)
x = x_tr2_end

# HPT
x_HPT_start = x
x_HPT_end   = x + L_HPT
draw_annulus(ax, x_HPT_start, x_HPT_end,
             HPT_inlet_tip, HPT_inlet_hub,
             HPT_outlet_tip, HPT_outlet_hub,
             COL_HPT, label='HPT')
x = x_HPT_end
x_end = x

# Shaft: centred on y=0, spans full engine length, drawn over everything
# at the centre but under annular fill via zorder
from matplotlib.patches import Rectangle
shaft_rect = Rectangle((x_inlet, -r_shaft), x_end - x_inlet, 2*r_shaft,
                        color=COL_SHAFT_FILL, zorder=3, linewidth=0)
ax.add_patch(shaft_rect)
ax.plot([x_inlet, x_end], [ r_shaft,  r_shaft], color=COL_SHAFT_EDGE, lw=1.0, zorder=4)
ax.plot([x_inlet, x_end], [-r_shaft, -r_shaft], color=COL_SHAFT_EDGE, lw=1.0, zorder=4)

# Centreline
ax.axhline(0, color='#555555', lw=0.8, linestyle='--', zorder=1, alpha=0.7)

# ---- Dimension arrows (above the engine) ----
y_top       = max(HPT_outlet_tip, D_lean/2)
y_dim_base  = y_top + 0.04
dim_arrow(ax, x_HPC_start, x_HPC_end,   y_dim_base,        f'HPC  {L_HPC*100:.0f} cm')
dim_arrow(ax, x_CC_start,  x_CC_end,    y_dim_base + 0.05, f'CC  {CC_L*100:.0f} cm')
dim_arrow(ax, x_HPT_start, x_HPT_end,   y_dim_base,        f'HPT  {L_HPT*100:.0f} cm')
dim_arrow(ax, x_inlet,     x_end,       y_dim_base + 0.10, f'Total  {(x_end-x_inlet)*100:.0f} cm')

# ---- Legend ----
legend_items = [
    mpatches.Patch(color=COL_HPC,       label=f'HPC  ({L_HPC*100:.0f} cm, {int(Stages_HPC)} stages)'),
    mpatches.Patch(color=COL_CC_RICH,   label=f'Rich zone  (D={D_rich*100:.0f} cm, L={L_rich*100:.0f} cm)'),
    mpatches.Patch(color=COL_CC_QUENCH, label=f'Quench zone  (L={L_quench*100:.0f} cm)'),
    mpatches.Patch(color=COL_CC_LEAN,   label=f'Lean zone  (D={D_lean*100:.0f} cm, L={L_lean*100:.0f} cm)'),
    mpatches.Patch(color=COL_HPT,       label=f'HPT  ({L_HPT*100:.0f} cm, {int(Stages_HPT)} stages)'),
    mpatches.Patch(color=COL_SHAFT_FILL,label=f'Shaft  (r={r_shaft*100:.1f} cm)'),
]
ax.legend(handles=legend_items, loc='upper right', fontsize=8,
          facecolor='#1a1a2e', edgecolor='#555555', labelcolor=COL_TEXT)

ax.set_xlabel('Axial position [m]', color=COL_TEXT, fontsize=10)
ax.set_ylabel('Radius [m]',         color=COL_TEXT, fontsize=10)
ax.set_title('Gas Turbine Cross-Section',
             color=COL_TEXT, fontsize=12, fontweight='bold', pad=12)
ax.tick_params(colors=COL_TEXT, labelsize=8)
for spine in ax.spines.values():
    spine.set_color(COL_GRID)
ax.set_xlim(x_inlet - 0.02, x_end + 0.02)
y_extent = max(HPT_outlet_tip, D_lean/2) + 0.14
ax.set_ylim(-y_extent, y_extent)
ax.grid(True, color=COL_GRID, lw=0.5, alpha=0.6)

plt.tight_layout()
plt.show()