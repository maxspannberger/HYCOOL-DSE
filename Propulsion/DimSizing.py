"""
DimSizing.py
============
Dimensional sizing of HPC, HPT, RQL combustor, and a system-level mass
estimate, given a converged thermodynamic cycle from TurbineSizing.

Class
-----
DimensionalSizing(engine, cfg)
    .size()         -> populates .results
    .report()       -> rich-formatted console output
    .plot()         -> to-scale cross-section diagram
    .to_csv_row()   -> flat dict for CSV export

Inputs come from `config.DimensionalConfig`. Tip / hub / Mach numbers,
mass anchors, RQL residence times and air fractions are all there.

Physics notes
-------------
HPT uses an "expand-inwards with minimal taper" rule:
  - tip is held constant at the HPT inlet value while the larger outlet
    annulus is swallowed by dropping the hub inward;
  - if the hub would dip below `HPT_Hub_Margin * HPC_inlet_hub`, the hub
    is pinned at that floor and the tip grows just enough to satisfy
    continuity. This mirrors the HPC's "compress inwards" geometry and
    keeps the rear of the engine narrow.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from CoolProp.CoolProp import PropsSI

from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich.columns import Columns
from rich import box

from Propulsion.TurbineSizing import GasTurbineCycle


# ----------------------------------------------------------------------
# Air property helpers
# ----------------------------------------------------------------------
def _air_gamma(p_bar, T):
    return (PropsSI('CPMASS', 'P', p_bar*1e5, 'T', T, 'Air') /
            PropsSI('CVMASS', 'P', p_bar*1e5, 'T', T, 'Air'))

def _air_cp(p_bar, T):
    return PropsSI('CPMASS', 'P', p_bar*1e5, 'T', T, 'Air')

def _air_rho(p_bar, T):
    R_air = 287.0
    return (p_bar * 1e5) / (R_air * T)


class DimensionalSizing:
    """
    Component-level geometry + mass estimate for a sized cycle.

    Parameters
    ----------
    engine : GasTurbineCycle
        A sized cycle. `engine.size()` must already have run.
    cfg : Config | DimensionalConfig
        Either a top-level Config or a DimensionalConfig instance.
    cea : optional CEA_Obj
        Required for the RQL combustor (stoichiometric O/F and Tflame).
        If None, get_Tcomb is skipped and the combustor uses fallback values.
    """

    R_air = 287.0

    def __init__(self, engine, cfg, cea=None):
        self.engine = engine
        self.dim    = cfg.dim if hasattr(cfg, "dim") else cfg
        self.cea    = cea
        self._console = Console()
        self.results  = None

    # ------------------------------------------------------------------
    # Build from config
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, engine, cfg, cea=None):
        return cls(engine=engine, cfg=cfg, cea=cea)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------
    def size(self):
        """Run all dimensional + mass calculations. Populates self.results."""
        if self.engine.results is None:
            raise RuntimeError("Pass a sized GasTurbineCycle (call .size() first).")

        d = self.dim
        results = self.engine.results
        design  = self.engine._design

        # ---- Station parameters ----
        P1, P2, Pc        = design["P1"], design["P2"], design["Pc"]
        P4, P5            = design["P4"], design["P5"]
        T1, T2, T2p       = design["T1"], design["T2"], design["T2p"]
        T4, T5            = design["T4"], design["T5"]
        mdot_f            = results["mdot_f"]
        mdot_air          = results["ideal_OF"] * mdot_f
        mdot_tot          = mdot_air + mdot_f

        # ===========================================================
        # HPC sizing
        # ===========================================================
        rho_1 = _air_rho(P1, T1)
        rho_2 = _air_rho(P2, T2)
        g_1   = _air_gamma(P1, T1)
        C_ax  = d.HPC_M_ax * np.sqrt(g_1 * self.R_air * T1)

        A_annulus        = mdot_air / (rho_1 * C_ax)
        A_annulus_outlet = mdot_air / (rho_2 * C_ax)

        inlet_tip  = np.sqrt(A_annulus        / (np.pi * (1 - d.HPC_Inlet_HTR**2)))
        inlet_hub  = d.HPC_Inlet_HTR  * inlet_tip
        outlet_tip = np.sqrt(A_annulus_outlet / (np.pi * (1 - d.HPC_Outlet_HTR**2)))
        outlet_hub = d.HPC_Outlet_HTR * outlet_tip

        r_mean_HPC_in  = 0.5 * (inlet_tip  + inlet_hub)
        r_mean_HPC_out = 0.5 * (outlet_tip + outlet_hub)
        r_mean_HPC     = 0.5 * (r_mean_HPC_in + r_mean_HPC_out)

        RPM_HPC    = (d.HPC_U_tip / inlet_tip) * (60 / (2 * np.pi))
        omega_HPC  = RPM_HPC * 2 * np.pi / 60
        U_mean_HPC = omega_HPC * r_mean_HPC

        Cp_hpc       = _air_cp(0.5*(P1 + P2), 0.5*(T1 + T2))
        Delta_h0_HPC = Cp_hpc * (T2 - T1)
        Stages_HPC   = int(np.ceil(Delta_h0_HPC / (d.HPC_Psi * U_mean_HPC**2)))
        L_HPC        = Stages_HPC * 2 * d.HPC_BladeChord * (1 + d.HPC_Spacing)

        # ===========================================================
        # HPT sizing (expand-inwards with taper fallback)
        # ===========================================================
        g_4      = _air_gamma(P4, T4)
        a_4      = np.sqrt(g_4 * self.R_air * T4)
        C_ax_HPT = d.HPT_M_ax * a_4

        rho_4 = _air_rho(P4, T4)
        rho_5 = _air_rho(P5, T5)

        A_HPT_inlet  = mdot_tot / (rho_4 * C_ax_HPT)
        A_HPT_outlet = mdot_tot / (rho_5 * C_ax_HPT)

        HPT_inlet_tip = np.sqrt(A_HPT_inlet / (np.pi * (1 - d.HPT_Inlet_HTR**2)))
        HPT_inlet_hub = d.HPT_Inlet_HTR * HPT_inlet_tip

        # Try constant-tip; fall back to pinned-hub if it would punch below the floor.
        r_hub_floor   = d.HPT_Hub_Margin * inlet_hub
        HPT_outlet_tip = HPT_inlet_tip
        r_hub_sq       = HPT_outlet_tip**2 - A_HPT_outlet / np.pi
        if r_hub_sq >= r_hub_floor**2:
            HPT_outlet_hub = np.sqrt(r_hub_sq)
        else:
            HPT_outlet_hub = r_hub_floor
            HPT_outlet_tip = np.sqrt(A_HPT_outlet / np.pi + HPT_outlet_hub**2)

        r_mean_HPT_in  = 0.5 * (HPT_inlet_tip  + HPT_inlet_hub)
        r_mean_HPT_out = 0.5 * (HPT_outlet_tip + HPT_outlet_hub)
        r_mean_HPT     = 0.5 * (r_mean_HPT_in  + r_mean_HPT_out)

        RPM_HPT    = (d.HPT_U_tip / HPT_inlet_tip) * (60 / (2 * np.pi))
        omega_HPT  = RPM_HPT * 2 * np.pi / 60
        U_mean_HPT = omega_HPT * r_mean_HPT

        Cp_hpt       = _air_cp(0.5*(P4 + P5), 0.5*(T4 + T5))
        Delta_h0_HPT = Cp_hpt * (T4 - T5)
        Stages_HPT   = int(np.ceil(Delta_h0_HPT / (d.HPT_Psi * U_mean_HPT**2)))
        L_HPT        = Stages_HPT * 2 * d.HPT_BladeChord * (1 + d.HPT_Spacing)

        # ===========================================================
        # RQL Combustor
        # ===========================================================
        # Stoichiometric O/F and flame temperature
        if self.cea is not None:
            OF_range    = np.arange(1, 200, 0.5)
            full_output = np.array([self.cea.get_Tcomb(Pc=Pc, MR=of) for of in OF_range])
            OF_stoich   = float(OF_range[np.argmax(full_output)])
            T_flame     = float(np.max(full_output))
        else:
            # Fallback values when CEA isn't available
            OF_stoich = 34.3
            T_flame   = 2400.0

        OF_total    = results["ideal_OF"]

        f_primary   = OF_stoich / OF_total
        f_secondary = 1.0 - f_primary - d.CC_f_quench

        mdot_primary = mdot_f * (1.0 + OF_stoich)
        mdot_quench  = mdot_primary + mdot_f * OF_total * d.CC_f_quench
        mdot_lean    = mdot_tot

        T_rich   = T_flame
        T_quench = 0.5 * (T_flame + T4)
        T_lean   = T4

        rho_rich   = Pc*1e5 / (self.R_air * T_rich)
        rho_quench = Pc*1e5 / (self.R_air * T_quench)
        rho_lean   = Pc*1e5 / (self.R_air * T_lean)

        A_rich   = mdot_primary / (rho_rich   * d.CC_C_ax)
        A_quench = mdot_quench  / (rho_quench * d.CC_C_ax)
        A_lean   = mdot_lean    / (rho_lean   * d.CC_C_ax)

        D_rich   = np.sqrt(4 * A_rich   / np.pi)
        D_quench = np.sqrt(4 * A_quench / np.pi)
        D_lean   = np.sqrt(4 * A_lean   / np.pi)

        V_rich = d.CC_tau_rich * mdot_primary / rho_rich
        V_lean = d.CC_tau_lean * mdot_lean    / rho_lean

        L_rich   = V_rich  / A_rich
        L_quench = 0.5 * D_quench
        L_lean   = V_lean  / A_lean
        CC_L     = L_rich + L_quench + L_lean

        # ===========================================================
        # Mass estimate
        # ===========================================================
        P_shaft_W      = results["total_net_W"]
        m_engine_total = P_shaft_W / d.SP_turboshaft

        m_ref_sum = d.m_HPC_kg_ref + d.m_CC_kg_ref + d.m_HPT_kg_ref
        f_HPC = d.m_HPC_kg_ref / m_ref_sum
        f_CC  = d.m_CC_kg_ref  / m_ref_sum
        f_HPT = d.m_HPT_kg_ref / m_ref_sum

        m_HPC = f_HPC * m_engine_total
        m_CC  = f_CC  * m_engine_total
        m_HPT = f_HPT * m_engine_total

        # Recuperator
        Cp_recup  = _air_cp(0.5*(P2 + Pc), 0.5*(T2p + T2))
        Q_recup_W = mdot_air * Cp_recup * abs(T2p - T2)
        m_recup   = Q_recup_W / d.SP_recup

        m_bare_subtotal = m_engine_total + m_recup
        m_system_margin = d.system_margin * m_bare_subtotal
        m_propulsion    = m_bare_subtotal + m_system_margin

        # ---- Stash everything ----
        self.results = dict(
            # Station passthrough
            P1=P1, P2=P2, Pc=Pc, P4=P4, P5=P5,
            T1=T1, T2=T2, T2p=T2p, T4=T4, T5=T5,
            mdot_f=mdot_f, mdot_air=mdot_air, mdot_tot=mdot_tot,
            # HPC
            HPC_inlet_tip=inlet_tip, HPC_inlet_hub=inlet_hub,
            HPC_outlet_tip=outlet_tip, HPC_outlet_hub=outlet_hub,
            HPC_RPM=RPM_HPC, HPC_U_mean=U_mean_HPC,
            HPC_Delta_h0=Delta_h0_HPC, HPC_Stages=Stages_HPC, HPC_L=L_HPC,
            HPC_A_in=A_annulus, HPC_A_out=A_annulus_outlet,
            HPC_rho_in=rho_1, HPC_rho_out=rho_2,
            # HPT
            HPT_inlet_tip=HPT_inlet_tip, HPT_inlet_hub=HPT_inlet_hub,
            HPT_outlet_tip=HPT_outlet_tip, HPT_outlet_hub=HPT_outlet_hub,
            HPT_RPM=RPM_HPT, HPT_U_mean=U_mean_HPT,
            HPT_Delta_h0=Delta_h0_HPT, HPT_Stages=Stages_HPT, HPT_L=L_HPT,
            HPT_A_in=A_HPT_inlet, HPT_A_out=A_HPT_outlet,
            HPT_rho_in=rho_4, HPT_rho_out=rho_5,
            # Combustor
            OF_stoich=OF_stoich, T_flame=T_flame, OF_total=OF_total,
            f_primary=f_primary, f_quench=d.CC_f_quench, f_secondary=f_secondary,
            D_rich=D_rich, D_quench=D_quench, D_lean=D_lean,
            L_rich=L_rich, L_quench=L_quench, L_lean=L_lean, CC_L=CC_L,
            # Mass
            m_engine_total=m_engine_total,
            m_HPC=m_HPC, m_CC=m_CC, m_HPT=m_HPT,
            Q_recup_W=Q_recup_W, m_recup=m_recup,
            m_bare_subtotal=m_bare_subtotal, m_system_margin=m_system_margin,
            m_propulsion=m_propulsion,
        )
        return self

    # ------------------------------------------------------------------
    # Flat dict for CSV
    # ------------------------------------------------------------------
    def to_csv_row(self):
        if self.results is None:
            raise RuntimeError("Call size() before to_csv_row().")
        return {f"dim__{k}": v for k, v in self.results.items()}

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    @staticmethod
    def _make_table(title, rows, color="magenta"):
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

    def report(self):
        if self.results is None:
            raise RuntimeError("Call size() before report().")
        r = self.results
        c = self._console

        # ------------- Mass --------------------------------------------
        c.print()
        c.rule("[bold white]PROPULSION SYSTEM MASS ESTIMATE[/bold white]")
        c.print()

        tbl_mass = self._make_table("Propulsion System Mass  (conceptual, ±30%)", [
            ("Shaft power",              f"{self.engine.results['total_net_W']/1e6:.3f}", "MW"),
            ("Specific power (anchor)",  f"{self.dim.SP_turboshaft/1e3:.1f}",              "kW/kg"),
            ("Bare engine mass (total)", f"{r['m_engine_total']:.1f}",                     "kg"),
            None,
            ("  HPC",                    f"{r['m_HPC']:.1f}", "kg"),
            ("  Combustor",              f"{r['m_CC']:.1f}",  "kg"),
            ("  HPT",                    f"{r['m_HPT']:.1f}", "kg"),
            None,
            ("Recuperator thermal duty", f"{r['Q_recup_W']/1e3:.1f}", "kW"),
            ("Recup. specific power",    f"{self.dim.SP_recup/1e3:.1f}", "kW/kg"),
            ("Recuperator mass",         f"{r['m_recup']:.1f}",       "kg"),
            None,
            ("Bare subtotal",            f"{r['m_bare_subtotal']:.1f}",   "kg"),
            (f"System margin ({self.dim.system_margin*100:.0f}%)",
                                         f"{r['m_system_margin']:.1f}",   "kg"),
            ("PROPULSION SYSTEM TOTAL",  f"{r['m_propulsion']:.1f}",      "kg"),
        ], color="magenta")
        c.print(tbl_mass)
        c.print()

        # ------------- Dimensional -------------------------------------
        c.rule("[bold white]COMPONENT DIMENSIONAL SIZING SUMMARY[/bold white]")
        c.print()

        tbl_HPC = self._make_table("HPC  (Axial, Air)", [
            ("Stages",            f"{r['HPC_Stages']:.0f}",            "-"),
            ("Length",            f"{r['HPC_L']*100:.1f}",             "cm"),
            None,
            ("Inlet tip radius",  f"{r['HPC_inlet_tip']*100:.2f}",     "cm"),
            ("Inlet hub radius",  f"{r['HPC_inlet_hub']*100:.2f}",     "cm"),
            ("Outlet tip radius", f"{r['HPC_outlet_tip']*100:.2f}",    "cm"),
            ("Outlet hub radius", f"{r['HPC_outlet_hub']*100:.2f}",    "cm"),
            None,
            ("RPM",               f"{r['HPC_RPM']:.0f}",               "rpm"),
            ("U_mean",            f"{r['HPC_U_mean']:.1f}",            "m/s"),
            ("Specific work",     f"{r['HPC_Delta_h0']/1e3:.1f}",      "kJ/kg"),
        ], color="blue")

        tbl_HPT = self._make_table("HPT  (Axial, Combustion Products)", [
            ("Stages",            f"{r['HPT_Stages']:.0f}",            "-"),
            ("Length",            f"{r['HPT_L']*100:.1f}",             "cm"),
            None,
            ("Inlet tip radius",  f"{r['HPT_inlet_tip']*100:.2f}",     "cm"),
            ("Inlet hub radius",  f"{r['HPT_inlet_hub']*100:.2f}",     "cm"),
            ("Outlet tip radius", f"{r['HPT_outlet_tip']*100:.2f}",    "cm"),
            ("Outlet hub radius", f"{r['HPT_outlet_hub']*100:.2f}",    "cm"),
            None,
            ("RPM",               f"{r['HPT_RPM']:.0f}",               "rpm"),
            ("U_mean",            f"{r['HPT_U_mean']:.1f}",            "m/s"),
            ("Specific work",     f"{r['HPT_Delta_h0']/1e3:.1f}",      "kJ/kg"),
        ], color="green")

        tbl_CC = self._make_table("Combustion Chamber  (RQL, H2/Air)", [
            ("Stoichiometric O/F", f"{r['OF_stoich']:.1f}", "-"),
            ("Adiabatic flame T",  f"{r['T_flame']:.0f}",   "K"),
            ("Overall O/F",        f"{r['OF_total']:.1f}",  "-"),
            None,
            ("Rich  D / L",        f"{r['D_rich']*100:.1f} / {r['L_rich']*100:.1f}",     "cm"),
            ("Quench D / L",       f"{r['D_quench']*100:.1f} / {r['L_quench']*100:.1f}", "cm"),
            ("Lean  D / L",        f"{r['D_lean']*100:.1f} / {r['L_lean']*100:.1f}",     "cm"),
            ("Total length",       f"{r['CC_L']*100:.1f}",  "cm"),
            None,
            ("f_primary",          f"{r['f_primary']:.3f}",   "-"),
            ("f_quench",           f"{r['f_quench']:.3f}",    "-"),
            ("f_secondary",        f"{r['f_secondary']:.3f}", "-"),
        ], color="red")

        c.print(Columns([tbl_HPC, tbl_HPT], equal=True, expand=False))
        c.print()
        c.print(tbl_CC)
        c.print()

        if r["f_secondary"] < 0:
            c.print(
                "[bold red]WARNING:[/bold red] f_secondary < 0: quench fraction "
                "too large for this O/F. Reduce CC_f_quench."
            )

    # ------------------------------------------------------------------
    # Cross-section diagram
    # ------------------------------------------------------------------
    def plot(self):
        if self.results is None:
            raise RuntimeError("Call size() before plot().")
        r = self.results

        # Geometry handles
        inlet_tip,  inlet_hub   = r["HPC_inlet_tip"],  r["HPC_inlet_hub"]
        outlet_tip, outlet_hub  = r["HPC_outlet_tip"], r["HPC_outlet_hub"]
        HPT_inlet_tip           = r["HPT_inlet_tip"]
        HPT_inlet_hub           = r["HPT_inlet_hub"]
        HPT_outlet_tip          = r["HPT_outlet_tip"]
        HPT_outlet_hub          = r["HPT_outlet_hub"]
        L_HPC, L_HPT            = r["HPC_L"], r["HPT_L"]
        L_rich, L_quench, L_lean = r["L_rich"], r["L_quench"], r["L_lean"]
        D_rich, D_quench, D_lean = r["D_rich"], r["D_quench"], r["D_lean"]
        CC_L                    = r["CC_L"]
        Stages_HPC, Stages_HPT  = r["HPC_Stages"], r["HPT_Stages"]

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_aspect('equal')
        ax.set_facecolor('#0d1117')
        fig.patch.set_facecolor('#0d1117')

        COL_HPC, COL_HPT       = '#4a9eff', '#4aff8a'
        COL_CC_RICH, COL_CC_QUENCH, COL_CC_LEAN = "#ff3535", "#ff9500", "#fff235"
        COL_CASING             = '#cccccc'
        COL_SHAFT_FILL, COL_SHAFT_EDGE = '#555555', '#999999'
        COL_TEXT, COL_GRID, COL_DIM = '#ffffff', '#2a2a2a', '#aaaaaa'

        r_shaft = min(inlet_hub, outlet_hub, HPT_inlet_hub, HPT_outlet_hub) * 0.85

        def draw_annulus(x0, x1, rt0, rh0, rt1, rh1, color, alpha=0.72, label=None):
            for s in (+1, -1):
                ax.fill([x0, x1, x1, x0],
                        [s*rt0, s*rt1, s*rh1, s*rh0],
                        color=color, alpha=alpha, zorder=4)
                ax.plot([x0, x1], [s*rt0, s*rt1], color=COL_CASING, lw=1.8, zorder=5)
                ax.plot([x0, x1], [s*rh0, s*rh1], color=COL_CASING, lw=1.2, zorder=5)
                ax.plot([x0, x0], [s*rh0, s*rt0], color=COL_CASING, lw=1.0, zorder=5)
                ax.plot([x1, x1], [s*rh1, s*rt1], color=COL_CASING, lw=1.0, zorder=5)
            if label:
                xm = 0.5*(x0 + x1)
                ym = 0.5*(0.5*(rt0+rh0) + 0.5*(rt1+rh1))
                ax.text(xm, ym, label, color=COL_TEXT, fontsize=9,
                        ha='center', va='center', fontweight='bold', zorder=7)

        def draw_combustor_zone(x0, x1, r0, r1, color, alpha=0.60, label=None):
            ax.fill([x0, x1, x1, x0], [r0, r1, -r1, -r0],
                    color=color, alpha=alpha, zorder=4)
            ax.plot([x0, x1], [ r0,  r1], color=COL_CASING, lw=2.0, zorder=5)
            ax.plot([x0, x1], [-r0, -r1], color=COL_CASING, lw=2.0, zorder=5)
            ax.plot([x0, x0], [-r0,  r0], color=COL_CASING, lw=1.0, zorder=5)
            ax.plot([x1, x1], [-r1,  r1], color=COL_CASING, lw=1.0, zorder=5)
            if label:
                ax.text(0.5*(x0+x1), 0.25*(r0+r1), label,
                        color=COL_TEXT, fontsize=9, ha='center', va='center',
                        fontweight='bold', zorder=7)

        def draw_transition(x0, x1, rt0, rh0, rt1, rh1):
            for s in (+1, -1):
                ax.plot([x0, x1], [s*rt0, s*rt1], color=COL_CASING, lw=1.5, zorder=5)
                ax.plot([x0, x1], [s*rh0, s*rh1], color=COL_CASING, lw=1.0,
                        linestyle=':', zorder=5)

        def dim_arrow(x1, x2, y, label):
            ax.annotate('', xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle='<->', color=COL_DIM, lw=1.2))
            ax.text(0.5*(x1+x2), y + 0.005, label, color=COL_DIM,
                    fontsize=7.5, ha='center', va='bottom', zorder=8)

        # ---- Layout ----
        x_inlet = 0.02
        x       = x_inlet
        x_HPC_start, x_HPC_end = x, x + L_HPC
        draw_annulus(x_HPC_start, x_HPC_end,
                     inlet_tip, inlet_hub, outlet_tip, outlet_hub,
                     COL_HPC, label='HPC')
        x = x_HPC_end

        L_trans_in = 0.04
        x_tr1_end  = x + L_trans_in
        draw_transition(x, x_tr1_end, outlet_tip, outlet_hub, D_rich/2, r_shaft)
        x = x_tr1_end

        x_CC_start   = x
        x_rich_end   = x + L_rich
        x_quench_end = x_rich_end + L_quench
        x_lean_end   = x_quench_end + L_lean
        draw_combustor_zone(x,            x_rich_end,   D_rich/2,   D_rich/2,   COL_CC_RICH,   "Rich")
        draw_combustor_zone(x_rich_end,   x_quench_end, D_rich/2,   D_quench/2, COL_CC_QUENCH, "Quench")
        draw_combustor_zone(x_quench_end, x_lean_end,   D_quench/2, D_lean/2,   COL_CC_LEAN,   "Lean")
        x = x_lean_end
        x_CC_end = x

        L_trans_out = 0.04
        x_tr2_end   = x + L_trans_out
        draw_transition(x, x_tr2_end, D_lean/2, r_shaft, HPT_inlet_tip, HPT_inlet_hub)
        x = x_tr2_end

        x_HPT_start, x_HPT_end = x, x + L_HPT
        draw_annulus(x_HPT_start, x_HPT_end,
                     HPT_inlet_tip, HPT_inlet_hub,
                     HPT_outlet_tip, HPT_outlet_hub,
                     COL_HPT, label='HPT')
        x_end = x_HPT_end

        shaft_rect = Rectangle((x_inlet, -r_shaft), x_end - x_inlet, 2*r_shaft,
                               color=COL_SHAFT_FILL, zorder=3, linewidth=0)
        ax.add_patch(shaft_rect)
        ax.plot([x_inlet, x_end], [ r_shaft,  r_shaft], color=COL_SHAFT_EDGE, lw=1.0, zorder=4)
        ax.plot([x_inlet, x_end], [-r_shaft, -r_shaft], color=COL_SHAFT_EDGE, lw=1.0, zorder=4)
        ax.axhline(0, color='#555555', lw=0.8, linestyle='--', zorder=1, alpha=0.7)

        y_top      = max(HPT_outlet_tip, D_lean/2)
        y_dim_base = y_top + 0.04
        dim_arrow(x_HPC_start, x_HPC_end, y_dim_base,        f'HPC  {L_HPC*100:.0f} cm')
        dim_arrow(x_CC_start,  x_CC_end,  y_dim_base + 0.05, f'CC  {CC_L*100:.0f} cm')
        dim_arrow(x_HPT_start, x_HPT_end, y_dim_base,        f'HPT  {L_HPT*100:.0f} cm')
        dim_arrow(x_inlet,     x_end,     y_dim_base + 0.10, f'Total  {(x_end-x_inlet)*100:.0f} cm')

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


# ----------------------------------------------------------------------
# Standalone smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from rocketcea.cea_obj_w_units import CEA_Obj
    from config import Config

    cfg    = Config()
    cea    = CEA_Obj(oxName="AIR", fuelName="GH2",
                     pressure_units="bar", temperature_units="K", isp_units="sec")
    engine = GasTurbineCycle.from_config(cfg).size(cea=cea)
    engine.report()

    dim = DimensionalSizing.from_config(engine, cfg, cea=cea).size()
    dim.report()
    dim.plot()
