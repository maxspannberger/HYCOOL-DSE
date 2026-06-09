from TurbineSizing import GasTurbineCycle
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich.columns import Columns
from rich.panel import Panel
from rich import box
from CoolProp.CoolProp import PropsSI

engine = GasTurbineCycle()
engine.size()

results = engine.results
design  = engine._design

_console = Console()

fluid   = engine.fluid
P_HC    = engine.PH1
T_HC    = engine.TH2

P_HD    = design["P3_H2"]

mdot    = results["mdot_f"]

# re-creating the thermal sizing stuff from TurbineSizing

h_HC    = PropsSI('H', 'P', P_HC*1e5, 'T', T_HC, fluid)  
s_HC    = PropsSI('S', 'P', P_HC*1e5, 'T', T_HC, fluid)   
rho_HC  = PropsSI('D', 'P', P_HC*1e5, 'T', T_HC, fluid)   
 
h_HD_s  = PropsSI('H', 'P', P_HD*1e5, 'S', s_HC, fluid)    
 
w_spec_exp  = (h_HC - h_HD_s) * engine.eta_H2T
h_HD_act    = h_HC - w_spec_exp                         
T_HD_act    = PropsSI('T', 'P', P_HD*1e5, 'H', h_HD_act, fluid)   
rho_HD      = PropsSI('D', 'P', P_HD*1e5, 'H', h_HD_act, fluid)   
 
# Density ratio = geometric expansion ratio required
# (ratio of outlet volume to inlet volume per unit mass)
# For a piston: this is approximately BDC volume / TDC volume (ignoring clearance)
rho_ratio   = rho_HC / rho_HD  

# Thermodynamic power output
P_exp_W     = w_spec_exp * mdot

V_dot_out   = mdot / rho_HD
V_dot_in    = mdot / rho_HC

f_crank     = 80                # Crank shaft frequency, higher = better tbh
N_cyl       = 4                 # Number of cylinders

V_swept     = V_dot_out / (f_crank * N_cyl)     # swept volume per cylinder per stroke

bore        = (4 * V_swept / np.pi)**(1.0/3.0)
stroke      = bore                              # Assuming square cylinder head

S_piston    = 2 * stroke * f_crank              # Mean velocity of piston head. Upper bound = 15m/s 
c_clearance = 1.0 / (rho_ratio - 1.0) if rho_ratio > 1.0 else 0.05      # <0.03 unreasonable. Clearance at top dead center

P_peak      = P_HC * 1e5
Mean_eff_p  = (P_exp_W / (f_crank * N_cyl)) / V_swept

sigma_allow = 600e6                             # Pa, Inconel-718 conservative estimate
r_bore      = bore / 2
t_wall      = (P_peak * r_bore) / (sigma_allow - 0.4 * P_peak)  # Minimum wall thickness

Sp_power    = 0.2                               # kW/kg, kind of a guess
m_exp       = Sp_power * (P_exp_W / 1e3)        # Guesstimated mass

# =====================================================================
# OUTPUT: GH2 Piston Expander Summary
# =====================================================================
 
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

_console.print()
_console.rule("[bold white]GH2 PISTON EXPANDER SIZING[/bold white]")
_console.print()
 
# Sanity check: cross-check expander power against TurbineSizing result
P_H2T_check = results["P_H2T_W"]                       # W, from TurbineSizing
 
tbl_exp_thermo = _make_table("GH2 Expander -- Thermodynamics  (ParaHydrogen, real gas)", [
    # -- Inlet conditions --
    ("Inlet pressure (HC)",      f"{P_HC:.0f}",               "bar"),
    ("Inlet temperature (HC)",   f"{T_HC:.0f}",               "K"),
    ("Inlet density",            f"{rho_HC:.3f}",             "kg/m\u00b3"),
    None,
    # -- Outlet conditions --
    ("Outlet pressure (HD)",     f"{P_HD:.1f}",               "bar"),
    ("Outlet temperature (HD)",  f"{T_HD_act:.1f}",           "K"),
    ("Outlet density",           f"{rho_HD:.4f}",             "kg/m\u00b3"),
    None,
    # -- Expansion summary --
    ("Pressure ratio",           f"{P_HC/P_HD:.1f}",          "-"),
    ("Density ratio (= vol ER)", f"{rho_ratio:.1f}",          "-"),
    ("Isentropic efficiency",    f"{engine.eta_H2T:.2f}",     "-"),
    ("Specific work (actual)",   f"{w_spec_exp/1e3:.1f}",     "kJ/kg"),
    None,
    # -- Power cross-check --
    ("H2 mass flow",             f"{mdot:.4f}",            "kg/s"),
    ("Expander power (calc)",    f"{P_exp_W/1e3:.2f}",        "kW"),
    ("Expander power (TurbSiz)", f"{P_H2T_check/1e3:.2f}",   "kW"),
], color="cyan")
 
tbl_exp_geom = _make_table("GH2 Expander -- Geometry  (Otto-cycle piston, single-acting)", [
    # -- Layout choices --
    ("Crankshaft frequency",     f"{f_crank:.0f}",            "Hz"),
    ("Number of cylinders",      f"{N_cyl:.0f}",              "-"),
    None,
    # -- Per-cylinder geometry --
    ("Swept volume per cyl",     f"{V_swept*1e6:.1f}",        "cc"),
    ("Bore",                     f"{bore*1e3:.1f}",           "mm"),
    ("Stroke",                   f"{stroke*1e3:.1f}",         "mm"),
    ("Wall thickness (Barlow)",  f"{t_wall*1e3:.1f}",         "mm"),
    None,
    # -- Performance indicators --
    ("Mean piston speed",        f"{S_piston:.2f}",           "m/s"),
    ("Mean effective pressure",  f"{Mean_eff_p/1e5:.2f}",            "bar"),
    ("Peak cylinder pressure",   f"{P_peak/1e5:.0f}",         "bar"),
    None,
    # -- Clearance check --
    ("Implied clearance frac",   f"{c_clearance:.3f}",        "-"),
    ("Vol flow at outlet",       f"{V_dot_out*1e3:.3f}",      "L/s"),
    ("Vol flow at inlet",        f"{V_dot_in*1e6:.2f}",       "cc/s"),
], color="yellow")
 
tbl_exp_mass = _make_table("GH2 Expander -- Mass Estimate", [
    ("Specific mass (mid-range)", f"{Sp_power:.1f}",       "kg/kW"),
    ("  range: 3-5 kg/kW based on H2 recip. compressor data", "", ""),
    ("  (Atlas Copco HX / Hofer HGD series)", "", ""),
    ("Expander power",           f"{P_exp_W/1e3:.2f}",        "kW"),
    ("Estimated assembly mass",  f"{m_exp:.1f}",              "kg"),
    None,
    ("NOTE: includes generator (50-80 Hz direct drive).", "", ""),
    ("Power converter for AC conditioning NOT included.", "", ""),
], color="magenta")
 
_console.print(Columns([tbl_exp_thermo, tbl_exp_geom], equal=False, expand=False))
_console.print()
_console.print(tbl_exp_mass)
_console.print()
 
# Warnings
if S_piston > 12.0:
    _console.print(f"[bold red]WARNING:[/bold red] Mean piston speed {S_piston:.1f} m/s exceeds 12 m/s. "
                   "Consider more cylinders or lower frequency.")
 
if c_clearance < 0.03:
    _console.print(f"[bold red]WARNING:[/bold red] Implied clearance fraction {c_clearance:.3f} < 0.03. "
                   "Geometric expansion ratio may be mechanically difficult to achieve in one stage. "
                   "Consider two-stage expansion or admission valve cutoff.")
 
if c_clearance > 0.15:
    _console.print(f"[bold yellow]NOTE:[/bold yellow] Clearance fraction {c_clearance:.3f} > 0.15. "
                   "Re-expansion losses will be non-trivial; refine with indicator diagram analysis.")
 
if abs(P_exp_W - P_H2T_check) / P_H2T_check > 0.01:
    _console.print(f"[bold red]WARNING:[/bold red] Expander power cross-check mismatch: "
                   f"calc={P_exp_W/1e3:.2f} kW vs TurbineSizing={P_H2T_check/1e3:.2f} kW. "
                   "Check eta_H2T consistency.")
 
_console.print("[dim]Sources:[/dim]")
_console.print("[dim]  Thermodynamics : CoolProp ParaHydrogen EOS (real gas)[/dim]")
_console.print("[dim]  Specific mass  : F1 Engines ~ 8 kW/kg using aero materials at 400 bar CC pressure[/dim]")
_console.print("[dim]                   4-6 kW/kg range taken as conservative bound)[/dim]")
_console.print("[dim]  Wall thickness : Barlow formula, IN718 sigma_allow=600 MPa[/dim]")
_console.print("[dim]  All values ±30% at conceptual design level[/dim]")
_console.print()