from rocketcea.cea_obj_w_units import CEA_Obj
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.rule import Rule

console = Console()

'''
Station Definitions:

P1 / T1 : Pre-HPC
P2 / T2 : Post-HPC / Pre-recuperator
T2'      : Post-recuperator / Pre-CC  (combustor air inlet)
P3 / T3 : CC
P4 / T4 : Post-CC / Pre-HPT
P5 / T5 : Post-HPT
----------------------------
PH1 / TH1: compressor
PH2 / TH2: Post-HEX / Pre-H2T
PH3 / TH3: Post-H2T

Exhaust cascade (single hot stream, two sinks in series):
    T5 -> [recuperator] and [H2 HEX] -> stack
    REGEN_FIRST sets which sink sees the hot exhaust first.
'''

# ─── Sizing Target ──────────────────────────────────────────────────────────
P_target    = 2.85e6       # W, net power target
TARGET_KEY  = "total_net_W"
mdot_f_init = 0.155        # kg/s, solver seed only

# ─── Cycle Parameters ─────────────────────────────────────────────────────────
P_ambient   = 0.38                      # bar, 25,000 ft
M0          = 0.7                       # Cruise Mach Number
PR_HPC      = 7

eta_HPC     = 0.90                      # Mattingly, HPC Efficiency
eta_CC      = 0.995                     # Combustor Efficiency
eta_CC_p    = 0.99                      # Combustor Pressure Efficiency
eta_HPT     = 0.90                      # Mattingly, HPT Efficiency
eta_HEX     = 0.90                      # Mattingly, HEX Efficiency
eta_mech    = 0.99                      # Mechanical efficiency
eta_diff    = 0.97                      # Diffuser Efficiency

Pc          = P_ambient * PR_HPC
PR_HPT      = 7
TIT         = 1500                      # K, turbine inlet temperature target
LHV_H2      = 120e6                     # J/kg, lower heating value (state convention)

# EFFICIENCY LEVER 1 -----------------------------------------------------------
# PR_HPT=11 expands only to ~1.04 bar while ambient is 0.38 bar, leaving pressure
# energy in the exhaust. FULL_EXPANSION expands to ambient (PR ~ Pc/P_ambient).
FULL_EXPANSION = True

# ─── Recuperator ───────────────────────────────────────────────────────────────
USE_REGEN   = True
eta_regen   = 0.775                 # McDonald, recuperator effectiveness
eta_regen_p = 0.95                  # recuperator pressure loss
REGEN_FIRST = False                 # False: H2 HEX takes the hot exhaust first, then regen
                                    # True : recuperator preheats air first, then H2 HEX
Regen_Fraction = 0.5                # 50% of the exhaust gets routed

# ─── Hydrogen Circuit Parameters ───────────────────────────────────────────────
fluid           = "ParaHydrogen"    # Ask Matthis
P_pre_comp      = 10
T_pre_comp      = 60
PH1             = 150
eta_compressor  = 0.7
eta_H2T     = 0.90                  # Mattingly, H2T efficiency
TH2         = 800                   # K, GH2 target temperature at HEX outlet


# ─── Air property helpers ──────────────────────────────────────────────────────
def air_gamma(p_bar, T):
    return (PropsSI('CPMASS', 'P', p_bar*1e5, 'T', T, 'Air') /
            PropsSI('CVMASS', 'P', p_bar*1e5, 'T', T, 'Air'))

def air_cp(p_bar, T):
    return PropsSI('CPMASS', 'P', p_bar*1e5, 'T', T, 'Air')

def air_h(p_bar, T):
    return PropsSI('H', 'P', p_bar*1e5, 'T', T, 'Air')


# ─── Design-Point Solve (everything independent of mdot_f) ─────────────────────
def solve_design(cea=None):
    """Solve all flow-independent quantities once.

    The recuperator couples the combustor air-inlet temperature to the O/F (more
    preheat -> leaner mixture), and with REGEN_FIRST=False that air-inlet temp
    also depends on the post-HEX exhaust, so O/F is found by a short fixed-point
    iteration. None of this depends on mdot_f, so the cycle stays linear in fuel
    flow and the sizing solver still lands in one step.
    """
    T0 = 239
    P0 = P_ambient

    g_0 = air_gamma(P0, T0)

    T1 = T0 * (1 + ((g_0 - 1) / 2) * M0**2)
    P1 = P0 * (1 + ((g_0 - 1) / 2) * M0**2)**(g_0 / (g_0 - 1)) * eta_diff

    # --- Compressor ---
    g_c = air_gamma(P1, T1)
    T2s = T1 * PR_HPC ** ((g_c - 1) / g_c)
    P2  = P1 * PR_HPC
    T2  = T1 + (T2s - T1) / eta_HPC
    Cp_HPC = air_cp(P2, 0.5*(T1+T2))

    # --- Combustor ---
    P2p = P2 * eta_regen_p if USE_REGEN else P2
    Pc = P2p

    # --- Turbine (full-expansion lever) ---
    PR_HPT_eff = (Pc / P_ambient) * eta_CC_p * (eta_HEX**2) if FULL_EXPANSION else PR_HPT
    g_t = air_gamma(Pc, TIT)
    T4  = TIT
    P4  = Pc * eta_CC_p
    T5s = T4 * (1 / PR_HPT_eff) ** ((g_t - 1) / g_t)
    P5  = Pc / PR_HPT_eff
    T5  = T4 - (T4 - T5s) * eta_HPT
    Cp_HPT = air_cp(P5, 0.5*(T4+T5))

    # --- Hydrogen compressor + HEX duty (per kg fuel, flow-independent) ---
    h1  = PropsSI("H", "P", P_pre_comp*1e5, "T", T_pre_comp, fluid)
    s1  = PropsSI("S", "P", P_pre_comp*1e5, "T", T_pre_comp, fluid)
    h2s = PropsSI("H", "P", PH1*1e5, "S", s1, fluid)
    h2  = h1 + (h2s - h1) / eta_compressor              # compressor outlet enthalpy
    w_compressor = h2 - h1                              # J per kg fuel
    h_hexout = PropsSI("H", "P", PH1*1e5, "T", TH2, fluid)
    dh_h2 = h_hexout - h2                               # HEX enthalpy rise, J per kg fuel

    cp_air_reg = air_cp(P2, 0.5*(T2 + T5))              # air-side cp across the recuperator

    # --- Fixed point: O/F <-> combustor air-inlet temp (recuperator coupling) ---
    def OF_for(T_air_in):
        # lean combustor energy balance; products approximated as air (NOTE: ~few
        # % off because of H2O in products; refine with CEA product cp if needed)
        return LHV_H2 * eta_CC / (air_h(Pc, TIT) - air_h(Pc, T_air_in))

    OF = OF_for(T2)
    T2p = T2
    T_hex_hot_in = T5
    T_exh_final = T5
    for _ in range(50):
        if USE_REGEN:
            r = OF * cp_air_reg / ((OF + 1) * Cp_HPT)     # C_air / C_exhaust

            if REGEN_FIRST:
                T_reg_in = T5
                T2p = T2 + eta_regen * (T_reg_in - T2)
                T_after_reg = T_reg_in - eta_regen * (T_reg_in - T2) * r
                T_hex_hot_in = T_after_reg
                T_exh_final = T_hex_hot_in - dh_h2 / ((OF + 1) * Cp_HPT)
            else:
                T_hex_hot_in = T5
                T_after_hex = T5 - dh_h2 / ((OF + 1) * Cp_HPT)
                T_reg_in = T_after_hex
                dpre = eta_regen * max(T_reg_in - T2, 0.0)
                T2p = T2 + dpre
                T_exh_final = T_reg_in - dpre * r
        else:
            T2p = T2
            T_hex_hot_in = T5
            T_exh_final = T5 - dh_h2 / (OF_for(T2) + 1)  # placeholder, recomputed below
        OF_new = OF_for(T2p)
        if abs(OF_new - OF) < 1e-9:
            OF = OF_new
            break
        OF = OF_new

    if not USE_REGEN:
        T_exh_final = T5 - dh_h2 / ((OF + 1) * Cp_HPT)

    Q_regen_per_mf = OF * cp_air_reg * (T2p - T2)   # recuperator duty, J per kg fuel

    # --- GH2 expander (per kg fuel) ---
    P3_H2 = Pc * 1.1  # 10% Margin for injector/combustion chamber pressure
    
    h2_in = PropsSI('H', 'P', PH1*1e5, 'T', TH2, fluid)
    s2_in = PropsSI('S', 'P', PH1*1e5, 'T', TH2, fluid)
    
    h3_ideal = PropsSI('H', 'P', P3_H2*1e5, 'S', s2_in, fluid)
    
    w_H2T = (h2_in - h3_ideal) * eta_H2T   # J per kg fuel
    h3_actual = h2_in - w_H2T                                               # Actual enthalpy
    
    TH3 = PropsSI('T', 'P', P3_H2*1e5, 'H', h3_actual, fluid)               # Exit temp
    
    Cp_H2T = (h2_in - h3_actual) / (TH2 - TH3) if TH2 != TH3 else 0

    # --- CEA molecular weight for Rs (reporting only) ---
    Rs = None
    if cea is not None:
        molwt, _ = cea.get_exit_MolWt_gamma(Pc=Pc, MR=OF, eps=1, frozen=1)
        Rs = 8.314 / (molwt / 1000)

    return {
        "T1": T1, "P1": P1, "T2": T2, "P2": P2, "T2p": T2p,
        "T4": T4, "P4": P4, "T5": T5, "P5": P5,
        "Cp_HPC": Cp_HPC, "Cp_HPT": Cp_HPT, "gamma": g_t, "PR_HPT_eff": PR_HPT_eff,
        "OF": OF, "Rs": Rs, "TIT": TIT,
        "h2_compressorout": h2, "w_compressor": w_compressor, "dh_h2": dh_h2,
        "w_H2T": w_H2T, "TH3": TH3, "Cp_H2T": Cp_H2T,
        "P3_H2": P3_H2, "h3_actual": h3_actual,   # NEW: needed by the H2 T-S plotters
        "T_hex_hot_in": T_hex_hot_in, "T_exh_final": T_exh_final,
        "Q_regen_per_mf": Q_regen_per_mf, "cp_air_reg": cp_air_reg,
    }




# ─── Full Cycle (scales the design point by mdot_f) ────────────────────────────
def run_cycle(mdot_f, d):
    mdot_tot = mdot_f * (d["OF"] + 1)


    P_HPC = d["Cp_HPC"] * mdot_f * (d["OF"]) * (d["T2"]  - d["T1"])
    P_HPT = d["Cp_HPT"] * mdot_f * (d["OF"]+1) * (d["T4"]  - d["T5"])
    P_H2T      = d["w_H2T"]  * mdot_f
    Power_compressor = d["w_compressor"] * mdot_f

    # --- HEX pinch sweep on the exhaust slice the HEX actually sees ---
    C_hot     = mdot_tot * d["Cp_HPT"]
    Q_tot     = d["dh_h2"] * mdot_f
    h_in      = d["h2_compressorout"]
    T_hot_in  = d["T_hex_hot_in"]
    T_hot_out = T_hot_in - Q_tot / C_hot

    N = 200
    q       = np.linspace(0, Q_tot, N)
    h_cold  = h_in + q / mdot_f
    T_cold  = np.array([PropsSI("T", "P", PH1*1e5, "H", h, fluid) for h in h_cold])
    T_hot   = T_hot_out + q / C_hot
    approach = T_hot - T_cold
    approach_min = approach.min()
    approach_loc = q[approach.argmin()] / Q_tot

    gaspath_net = P_HPT - P_HPC
    h2_net      = P_H2T - Power_compressor
    total_net   = (P_HPT + P_H2T - P_HPC - Power_compressor)*eta_mech

    q_in = mdot_f * LHV_H2
    eta_total   = total_net   / q_in
    eta_gaspath = gaspath_net / q_in

    return {
        "mdot_f": mdot_f, "mdot_tot": mdot_tot, "ideal_OF": d["OF"],
        "Rs_ideal": d["Rs"], "gamma_ideal": d["gamma"], "TIT": d["TIT"],
        "T1": d["T1"], "P1": d["P1"], "T2": d["T2"], "P2": d["P2"],
        "T2p": d["T2p"], "T3": d["TIT"], "P3": Pc, "T4": d["T4"], "P4": d["P4"],
        "T5": d["T5"], "P5": d["P5"],
        "Cp_HPC": d["Cp_HPC"], "P_HPC_W": P_HPC,
        "Cp_HPT": d["Cp_HPT"], "P_HPT_W": P_HPT,
        "gaspath_net_W": gaspath_net,
        "Q_tot_W": Q_tot, "T_hex_hot_in": T_hot_in, "T_hot_out": T_hot_out,
        "T_exh_final": d["T_exh_final"],
        "approach_min": approach_min, "approach_loc": approach_loc,
        "hex_feasible": approach_min > 0,
        "q": q, "T_cold": T_cold, "T_hot": T_hot, "approach": approach,
        "P_H2T_W": P_H2T, "Power_compressor_W": Power_compressor, "h2_net_W": h2_net,
        "Q_regen_W": d["Q_regen_per_mf"] * mdot_f,
        "total_net_W": total_net,
        "q_in_W": q_in, "eta_total": eta_total, "eta_gaspath": eta_gaspath,
        # NEW: hydrogen state data propagated through so the T-S plotters can reach it
        "h2_compressorout": d["h2_compressorout"], "P3_H2": d["P3_H2"],
        "h3_actual": d["h3_actual"], "TH3": d["TH3"],
    }


# ─── Fuel-Flow Sizing (still one secant step: linear in mdot_f) ────────────────
def size_fuel_flow(P_target, d, target_key=TARGET_KEY,
                   mdot_f_guess=mdot_f_init, tol=1.0, max_iter=50):
    history = []

    def residual(mdot_f):
        res = run_cycle(mdot_f, d)
        return res[target_key] - P_target, res

    x0 = mdot_f_guess
    x1 = mdot_f_guess * 1.1
    f0, _   = residual(x0)
    f1, res = residual(x1)
    history.append((x0, f0)); history.append((x1, f1))
    for _ in range(max_iter):
        if abs(f1) <= tol or f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if x2 <= 0:
            x2 = 0.5 * x1
        f2, res = residual(x2)
        history.append((x2, f2))
        x0, f0, x1, f1 = x1, f1, x2, f2
    return x1, res, history

# ─── Reporting ─────────────────────────────────────────────────────────────────
def report_all(P_target, mdot_f, history, res):
    """Consolidated terminal reporting for the entire engine cycle."""
    console.print(Rule("[bold cyan]Combustion / Fuel Balance[/bold cyan]"))
    
    t_comb = Table(box=box.ROUNDED, header_style="bold magenta", show_lines=True)
    t_comb.add_column("Parameter", style="cyan", justify="left", min_width=34)
    t_comb.add_column("Value", style="yellow", justify="right", min_width=14)
    t_comb.add_column("Units", style="dim", justify="left")
    t_comb.add_row("Turbine Inlet Temperature", f"{res['TIT']:.1f}", "K")
    t_comb.add_row("Combustor air inlet (T2')", f"{res['T2p']:.1f}", "K")
    t_comb.add_row("O/F (energy balance at T2')", f"{res['ideal_OF']:.2f}", "-")
    t_comb.add_row("Gamma (air, at T3)", f"{res['gamma_ideal']:.4f}", "-")
    if res["Rs_ideal"] is not None:
        t_comb.add_row("Specific Gas Constant (CEA)", f"{res['Rs_ideal']:.2f}", "J/(kg·K)")
    t_comb.add_row("Total Mass-Flow Rate", f"{res['mdot_tot']:.4f}", "kg/s")
    console.print(t_comb)

    console.print(Rule("[bold cyan]Fuel-Flow Sizing[/bold cyan]"))
    t_size = Table(box=box.ROUNDED, header_style="bold green", show_lines=True)
    t_size.add_column("Iter", style="bold white", justify="center")
    t_size.add_column("mdot_f [kg/s]", style="yellow", justify="right")
    t_size.add_column("Residual [MW]", style="cyan", justify="right")
    for i, (m, f) in enumerate(history):
        t_size.add_row(str(i), f"{m:.6f}", f"{f/1e6:+.4f}")
    console.print(t_size)
    console.print(Panel(
        f"Target net power: [bold]{P_target/1e6:.3f} MW[/bold]\n"
        f"Sized fuel flow:  [bold green]{mdot_f:.6f} kg/s[/bold green]",
        border_style="green"))

    if USE_REGEN:
        console.print(Rule("[bold cyan]Recuperator[/bold cyan]"))
        t_reg = Table(box=box.ROUNDED, header_style="bold yellow", show_lines=True)
        t_reg.add_column("Parameter", style="cyan", justify="left", min_width=32)
        t_reg.add_column("Value", style="yellow", justify="right", min_width=14)
        t_reg.add_column("Units", style="dim", justify="left")
        t_reg.add_row("Effectiveness", f"{eta_regen:.2f}", "-")
        t_reg.add_row("Air in (T2)", f"{res['T2']:.1f}", "K")
        t_reg.add_row("Air out (T2')", f"{res['T2p']:.1f}", "K")
        t_reg.add_row("Recuperator duty", f"{res['Q_regen_W']/1e6:.3f}", "MW")
        t_reg.add_row("Final stack exhaust", f"{res['T_exh_final']:.1f}", "K")
        console.print(t_reg)

    console.print(Rule("[bold cyan]Thermodynamic Cycle & Power[/bold cyan]"))
    t_stat = Table(box=box.SIMPLE_HEAVY, header_style="bold blue", show_lines=True)
    t_stat.add_column("Station", style="bold white", justify="center")
    t_stat.add_column("Description", style="cyan", justify="left")
    t_stat.add_column("T [K]", style="yellow", justify="right")
    t_stat.add_column("P [bar]", style="green", justify="right")
    t_stat.add_row("1", "Pre-HPC", f"{res['T1']:.1f}", f"{res['P1']:.3f}")
    t_stat.add_row("2", "Post-HPC", f"{res['T2']:.1f}", f"{res['P2']:.3f}")
    t_stat.add_row("2'", "Post-recuperator", f"{res['T2p']:.1f}", f"{res['P2']:.3f}")
    t_stat.add_row("3", "Combustion Chamber", f"{res['T3']:.1f}", f"{res['P3']:.3f}")
    t_stat.add_row("4", "Post-CC / Pre-HPT", f"{res['T4']:.1f}", f"{res['P4']:.3f}")
    t_stat.add_row("5", "Post-HPT", f"{res['T5']:.1f}", f"{res['P5']:.3f}")
    console.print(t_stat)

    console.print(Rule("[bold cyan]Hydrogen HEX & Expander Turbine[/bold cyan]"))
    t_hex = Table(box=box.ROUNDED, header_style="bold yellow", show_lines=True)
    t_hex.add_column("Parameter", style="cyan", justify="left", min_width=32)
    t_hex.add_column("Value", style="yellow", justify="right", min_width=14)
    t_hex.add_column("Units", style="dim", justify="left")
    t_hex.add_row("Hot-side inlet to HEX", f"{res['T_hex_hot_in']:.1f}", "K")
    t_hex.add_row("Total heat transferred", f"{res['Q_tot_W']/1e6:.3f}", "MW")
    t_hex.add_row("Exhaust temp post-HEX", f"{res['T_hot_out']:.1f}", "K")
    t_hex.add_row("Minimum approach temp", f"{res['approach_min']:.1f}", "K")
    console.print(t_hex)
    
    t_h2t = Table(box=box.ROUNDED, header_style="bold magenta", show_lines=True)
    t_h2t.add_column("Component", style="cyan", justify="left", min_width=30)
    t_h2t.add_column("Power [kW]", style="yellow", justify="right", min_width=14)
    t_h2t.add_row("GH2 Turbine (produced)", f"[green]+{res['P_H2T_W']/1e3:.2f}[/green]")
    t_h2t.add_row("compressor (consumed)", f"[red]-{res['Power_compressor_W']/1e3:.2f}[/red]")
    net_h2 = res["h2_net_W"] / 1e3
    nc = "green" if net_h2 >= 0 else "red"
    t_h2t.add_row("[bold]Net GH2 circuit[/bold]", f"[bold {nc}]{net_h2:+.2f}[/bold {nc}]")
    console.print(t_h2t)

    console.print(Rule("[bold cyan]Overall Efficiency & Power Summary[/bold cyan]"))
    t_sum = Table(box=box.DOUBLE_EDGE, header_style="bold white", show_lines=True)
    t_sum.add_column("Source", style="cyan", justify="left", min_width=28)
    t_sum.add_column("Value", style="yellow", justify="right", min_width=14)
    t_sum.add_row("Fuel power in (LHV)", f"{res['q_in_W']/1e6:.3f} MW")
    t_sum.add_row("Gas-path efficiency", f"{res['eta_gaspath']*100:.1f} %")
    t_sum.add_row("Total Efficiency", f"[bold green]{res['eta_total']*100:.1f} %[/bold green]")
    t_sum.add_row("HPT output", f"[green]+{res['P_HPT_W']/1e6:.3f} MW[/green]")
    t_sum.add_row("GH2 turbine output", f"[green]+{res['P_H2T_W']/1e6:.3f} MW[/green]")
    t_sum.add_row("HPC demand", f"[red]-{res['P_HPC_W']/1e6:.3f} MW[/red]")
    t_sum.add_row("compressor demand", f"[red]-{res['Power_compressor_W']/1e6:.3f} MW[/red]")
    total = res["total_net_W"] / 1e6
    tc = "green" if total >= 0 else "red"
    t_sum.add_row("[bold]TOTAL NET SHAFT[/bold]", f"[bold {tc}]{total:+.3f} MW[/bold {tc}]")
    console.print(t_sum)


# ─── T-S plotting helpers (shared) ───────────────────────────────────────────────
def _isobar(P_bar, T_start, T_end, fluid_name, num_points=60):
    """Real isobaric s-T curve between two temperatures for a given fluid."""
    if abs(T_start - T_end) < 0.1:
        return np.array([]), np.array([])
    T_arr = np.linspace(T_start, T_end, num_points)
    S_arr = np.array([PropsSI('S', 'P', P_bar*1e5, 'T', t, fluid_name) for t in T_arr])
    return S_arr, T_arr


def h2_state_points(res):
    """Return the four hydrogen state points as (name, P_bar, T, s).

    HA : compressor inlet  (tank-side feed, supercritical at T_pre_comp)
    HB : compressor outlet  (PH1)
    HC : HEX outlet / turbine inlet  (PH1, TH2)
    HD : turbine outlet  (P3_H2, TH3) -> on to the combustor (open path, not a loop)
    """
    PA, TA = P_pre_comp, T_pre_comp
    sA = PropsSI('S', 'P', PA*1e5, 'T', TA, fluid)

    PB = PH1
    hB = res['h2_compressorout']
    TB = PropsSI('T', 'P', PB*1e5, 'H', hB, fluid)
    sB = PropsSI('S', 'P', PB*1e5, 'H', hB, fluid)

    PC, TC = PH1, TH2
    sC = PropsSI('S', 'P', PC*1e5, 'T', TC, fluid)

    PD = res['P3_H2']
    hD = res['h3_actual']
    TD = res['TH3']
    sD = PropsSI('S', 'P', PD*1e5, 'H', hD, fluid)

    return [("A", PA, TA, sA), ("B", PB, TB, sB),
            ("C", PC, TC, sC), ("D", PD, TD, sD)]


def _draw_h2_path(ax, states, label_prefix=""):
    """Draw the open hydrogen path HA->HB->HC->HD on a given axis."""
    (nA, PA, TA, sA), (nB, PB, TB, sB), (nC, PC, TC, sC), (nD, PD, TD, sD) = states

    # HA -> HB : compression (tie-line; the process is not isobaric)
    ax.plot([sA, sB], [TA, TB], '-', color='steelblue', linewidth=2,
            label=f"{label_prefix}H2 compression")
    # HB -> HC : HEX heat addition (real isobar at PH1)
    S_hex, T_hex = _isobar(PB, TB, TC, fluid)
    if S_hex.size:
        ax.plot(S_hex, T_hex, '-', color='crimson', linewidth=2.5,
                label=f"{label_prefix}H2 HEX heating")
    # HC -> HD : expander turbine (tie-line)
    ax.plot([sC, sD], [TC, TD], '-', color='seagreen', linewidth=2,
            label=f"{label_prefix}H2 turbine")

    for name, P, T, s in states:
        ax.plot(s, T, 'ko', markersize=5, zorder=5)
        ax.annotate(f" H{name}", (s, T), fontsize=10, xytext=(5, 2),
                    textcoords="offset points")
    # the path is open: fuel leaves the turbine and enters the combustor (no return to HA)
    ax.annotate("to combustor", (sD, TD), fontsize=8, color='seagreen',
                xytext=(8, -12), textcoords="offset points")


def report_plots(res):
    """Generates a highly detailed T-S diagram for the cycle."""
    plt.figure(figsize=(9, 7))
    
    def calc_isobar(P_bar, T_start, T_end, num_points=50):
        if abs(T_start - T_end) < 0.1:
            return [], [] 
        T_arr = np.linspace(T_start, T_end, num_points)
        S_arr = [PropsSI('S', 'P', P_bar * 1e5, 'T', t, 'Air') for t in T_arr]
        return S_arr, T_arr

    # Extract state points
    P1, T1 = res['P1'], res['T1']
    P2, T2, T2p = res['P2'], res['T2'], res['T2p']
    P4, T4 = res['P4'], res['T4']
    P5, T5 = res['P5'], res['T5']
    T_exh = res['T_exh_final']

    s1 = PropsSI('S', 'P', P1 * 1e5, 'T', T1, 'Air')
    s2 = PropsSI('S', 'P', P2 * 1e5, 'T', T2, 'Air')
    s2p = PropsSI('S', 'P', P2 * 1e5, 'T', T2p, 'Air')
    s4 = PropsSI('S', 'P', P4 * 1e5, 'T', T4, 'Air')
    s5 = PropsSI('S', 'P', P5 * 1e5, 'T', T5, 'Air')
    s_exh = PropsSI('S', 'P', P5 * 1e5, 'T', T_exh, 'Air')

    # 1 -> 2: Compression
    plt.plot([s1, s2], [T1, T2], 'k-', linewidth=1.5, label='Compression / Expansion')
    
    # 2 -> 2': Regenerator Heating (Exact Isobar)
    S_reg_cold, T_reg_cold = calc_isobar(P2, T2, T2p)
    if S_reg_cold:
        plt.plot(S_reg_cold, T_reg_cold, color='orange', linewidth=3, label="Regenerator (Air Heating)")
        
    # 2' -> 4: Combustor (Exact Isobar)
    S_comb, T_comb = calc_isobar(P4, T2p, T4)
    plt.plot(S_comb, T_comb, color='red', linewidth=2, label="Combustor (Heat Addition)")
    
    # 4 -> 5: Expansion
    plt.plot([s4, s5], [T4, T5], 'k-', linewidth=1.5)
    
    # 5 -> T_exh: Exhaust Cooling (Exact Isobar - feeds Regen and HEX)
    S_exh, T_exh_arr = calc_isobar(P5, T5, T_exh)
    plt.plot(S_exh, T_exh_arr, color='purple', linewidth=3, label="Exhaust (Regen + HEX Sink)")

    # T_exh -> 1: Atmospheric Heat Rejection 
    # Replaced isobar function with a direct tie-line to perfectly close the thermodynamic loop
    # handling the mixing and pressure drop from P5 back to P1.
    plt.plot([s_exh, s1], [T_exh, T1], color='gray', linestyle='--', linewidth=1.5, label="Atmospheric Rejection")

    # Visual linkage mapping the heat transfer
    if T2p > T2:
        T_cold_mid = (T2 + T2p) / 2
        s_cold_mid = PropsSI('S', 'P', P2 * 1e5, 'T', T_cold_mid, 'Air')
        
        # Calculate matching thermal centroid on the hot stream
        T_hot_mid = T5 - (T2p - T2) / 2  
        s_hot_mid = PropsSI('S', 'P', P5 * 1e5, 'T', T_hot_mid, 'Air')
        
        plt.annotate('', xy=(s_cold_mid, T_cold_mid), xytext=(s_hot_mid, T_hot_mid),
                     arrowprops=dict(arrowstyle='->', color='orange', linestyle='-', lw=2))
        
        plt.text((s_cold_mid + s_hot_mid)/2, (T_cold_mid + T_hot_mid)/2 + 25, 'Regen Heat Transfer', 
                 color='orange', ha='center', fontsize=9, fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Station markers
    stations = {"1": (s1, T1), "2": (s2, T2), "2'": (s2p, T2p), 
                "4": (s4, T4), "5": (s5, T5), "Exh": (s_exh, T_exh)}
    
    for name, (s, t) in stations.items():
        if name == "2'" and T2p == T2: 
            continue 
        plt.plot(s, t, 'ko', markersize=5, zorder=5)
        plt.annotate(f" {name}", (s, t), fontsize=10, xytext=(5, 2), textcoords="offset points")

    plt.title("Gas-Path T-S Diagram")
    plt.xlabel("Specific Entropy, s [J/kg·K]")
    plt.ylabel("Temperature, T [K]")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()


def report_plots_h2(res):
    """Standalone hydrogen-circuit T-S diagram (ParaHydrogen working fluid).

    This is the physically correct view of the H2 side: the entropy scale and
    reference state are hydrogen's own, so the numbers here are NOT comparable to
    the air diagram. Path is open (HA->HB->HC->HD, then fuel enters the combustor).
    """
    states = h2_state_points(res)
    plt.figure(figsize=(9, 7))
    _draw_h2_path(plt.gca(), states)
    plt.title("Hydrogen-Circuit T-S Diagram (ParaHydrogen)")
    plt.xlabel("Specific Entropy, s [J/kg·K]  (hydrogen reference)")
    plt.ylabel("Temperature, T [K]")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def report_plots_overlay(res):
    """Air gas path and hydrogen circuit on shared T but INDEPENDENT s axes.

    The two working fluids have different entropy reference states and live on
    different numerical scales (air ~ 4-5 kJ/kg.K, H2 ~ 24-56 kJ/kg.K). A single
    shared s-axis would be physically meaningless, so hydrogen gets its own top
    x-axis via twiny(). Only the temperature (y) axis is common and comparable.
    """
    fig, ax_air = plt.subplots(figsize=(11, 7))

    # ---- Air gas path on the bottom axis ----
    P1, T1 = res['P1'], res['T1']
    P2, T2, T2p = res['P2'], res['T2'], res['T2p']
    P4, T4 = res['P4'], res['T4']
    P5, T5 = res['P5'], res['T5']
    T_exh = res['T_exh_final']

    s1 = PropsSI('S', 'P', P1*1e5, 'T', T1, 'Air')
    s2 = PropsSI('S', 'P', P2*1e5, 'T', T2, 'Air')
    s2p = PropsSI('S', 'P', P2*1e5, 'T', T2p, 'Air')
    s4 = PropsSI('S', 'P', P4*1e5, 'T', T4, 'Air')
    s5 = PropsSI('S', 'P', P5*1e5, 'T', T5, 'Air')
    s_exh = PropsSI('S', 'P', P5*1e5, 'T', T_exh, 'Air')

    ax_air.plot([s1, s2], [T1, T2], 'k-', lw=1.5, label="Air compression / expansion")
    Sr, Tr = _isobar(P2, T2, T2p, 'Air')
    if Sr.size:
        ax_air.plot(Sr, Tr, color='orange', lw=3, label="Air recuperator heating")
    Sc, Tc = _isobar(P4, T2p, T4, 'Air')
    ax_air.plot(Sc, Tc, color='red', lw=2, label="Air combustor")
    ax_air.plot([s4, s5], [T4, T5], 'k-', lw=1.5)
    Se, Te = _isobar(P5, T5, T_exh, 'Air')
    ax_air.plot(Se, Te, color='purple', lw=3, label="Air exhaust (regen + HEX)")
    ax_air.plot([s_exh, s1], [T_exh, T1], color='gray', ls='--', lw=1.5, label="Atmospheric rejection")
    for name, s, t in [("1", s1, T1), ("2", s2, T2), ("2'", s2p, T2p),
                       ("4", s4, T4), ("5", s5, T5), ("Exh", s_exh, T_exh)]:
        if name == "2'" and T2p == T2:
            continue
        ax_air.plot(s, t, 'ko', ms=5, zorder=5)
        ax_air.annotate(f" {name}", (s, t), fontsize=9, xytext=(4, 2), textcoords="offset points")

    ax_air.set_xlabel("Air specific entropy, s [J/kg·K]  (air reference)", color='black')
    ax_air.set_ylabel("Temperature, T [K]")
    ax_air.grid(True, alpha=0.4)

    # ---- Hydrogen circuit on the top axis (independent entropy scale) ----
    ax_h2 = ax_air.twiny()
    states = h2_state_points(res)
    _draw_h2_path(ax_h2, states)
    ax_h2.set_xlabel("Hydrogen specific entropy, s [J/kg·K]  (hydrogen reference - NOT comparable to air)",
                     color='dimgray')

    ax_air.set_title("Overlay: Air Gas Path + Hydrogen Circuit\n"
                     "(shared temperature axis; the two entropy axes are independent)")

    # merge legends from both axes into one box
    h1, l1 = ax_air.get_legend_handles_labels()
    h2, l2 = ax_h2.get_legend_handles_labels()
    ax_air.legend(h1 + h2, l1 + l2, loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0.)
    fig.tight_layout()
    plt.show()

# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cea = CEA_Obj(oxName="AIR", fuelName="GH2",
                  pressure_units="bar", temperature_units="K", isp_units="sec")
    d = solve_design(cea)
    mdot_f, res, history = size_fuel_flow(P_target, d)

    report_all(P_target, mdot_f, history, res)
    report_plots(res)          # air gas path (unchanged)
    report_plots_h2(res)       # NEW: standalone hydrogen circuit
    report_plots_overlay(res)  # NEW: both fluids, shared T, independent s axes