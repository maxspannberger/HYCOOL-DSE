"""
ExpanderSizing.py
=================
GH2 piston expander sizing for the hydrogen fuel circuit.

The piston expander sits between the H2 HEX (which warmed the GH2 up to
TH2 at PH1) and the combustor injector (PD = P3_H2). It extracts shaft
power from the pressure drop and is the source of the `w_H2T` term in
the cycle's net-power budget.

Class
-----
PistonExpander(engine, cfg)
    .size()       -> populates .results
    .report()     -> rich-formatted console output
    .to_csv_row() -> flat dict for CSV export
"""

import numpy as np
from CoolProp.CoolProp import PropsSI

from rich.console import Console
from rich.table import Table
from rich.columns import Columns
from rich import box

from Propulsion.TurbineSizing import GasTurbineCycle


class PistonExpander:
    """
    Otto-cycle piston expander on the GH2 circuit (single-acting).

    Parameters
    ----------
    engine : GasTurbineCycle
        A sized cycle (engine.size() must already have run).
    cfg : Config | ExpanderConfig
        Either the full Config or a bare ExpanderConfig.
    """

    def __init__(self, engine, cfg):
        self.engine = engine
        self.exp    = cfg.expander if hasattr(cfg, "expander") else cfg
        self._console = Console()
        self.results  = None

    @classmethod
    def from_config(cls, engine, cfg):
        return cls(engine=engine, cfg=cfg)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def size(self):
        if self.engine.results is None:
            raise RuntimeError("Pass a sized GasTurbineCycle (call .size() first).")

        e       = self.engine
        results = e.results
        design  = e._design
        ex      = self.exp

        fluid   = e.fluid
        P_HC    = e.PH1
        T_HC    = design["TH2"]   # TH2 is now dynamic (set in TurbineSizing)
        P_HD    = design["P3_H2"]
        mdot    = results["mdot_f"]

        # ---- Expansion thermodynamics ----
        h_HC   = PropsSI('H', 'P', P_HC*1e5, 'T', T_HC, fluid)
        s_HC   = PropsSI('S', 'P', P_HC*1e5, 'T', T_HC, fluid)
        rho_HC = PropsSI('D', 'P', P_HC*1e5, 'T', T_HC, fluid)
        h_HD_s = PropsSI('H', 'P', P_HD*1e5, 'S', s_HC, fluid)

        w_spec_exp = (h_HC - h_HD_s) * e.eta_H2T
        h_HD_act   = h_HC - w_spec_exp
        T_HD_act   = PropsSI('T', 'P', P_HD*1e5, 'H', h_HD_act, fluid)
        rho_HD     = PropsSI('D', 'P', P_HD*1e5, 'H', h_HD_act, fluid)

        # Density ratio = required geometric expansion ratio
        rho_ratio = rho_HC / rho_HD
        P_exp_W   = w_spec_exp * mdot

        V_dot_out = mdot / rho_HD
        V_dot_in  = mdot / rho_HC

        # ---- Geometry (single-acting Otto-cycle piston) ----
        V_swept     = V_dot_out / (ex.f_crank * ex.N_cyl)
        bore        = (4 * V_swept / np.pi) ** (1.0/3.0)
        stroke      = bore
        S_piston    = 2 * stroke * ex.f_crank
        c_clearance = 1.0 / (rho_ratio - 1.0) if rho_ratio > 1.0 else 0.05

        P_peak     = P_HC * 1e5
        Mean_eff_p = (P_exp_W / (ex.f_crank * ex.N_cyl)) / V_swept

        # ---- Wall thickness via Barlow ----
        r_bore = bore / 2
        t_wall = (P_peak * r_bore) / (ex.sigma_allow - 0.4 * P_peak)

        # ---- Mass estimate ----
        m_exp = ex.Sp_power * (P_exp_W / 1e3)

        # ---- Cross-check against cycle ----
        P_H2T_cycle = results["P_H2T_W"]
        power_mismatch = abs(P_exp_W - P_H2T_cycle) / P_H2T_cycle if P_H2T_cycle else 0.0

        self.results = dict(
            P_HC=P_HC, T_HC=T_HC, rho_HC=rho_HC,
            P_HD=P_HD, T_HD=T_HD_act, rho_HD=rho_HD,
            PR=P_HC/P_HD, rho_ratio=rho_ratio,
            eta_H2T=e.eta_H2T, w_spec_exp=w_spec_exp,
            mdot=mdot, P_exp_W=P_exp_W,
            f_crank=ex.f_crank, N_cyl=ex.N_cyl,
            V_swept=V_swept, bore=bore, stroke=stroke,
            S_piston=S_piston, c_clearance=c_clearance,
            P_peak=P_peak, Mean_eff_p=Mean_eff_p,
            t_wall=t_wall,
            Sp_power=ex.Sp_power, m_exp=m_exp,
            P_H2T_cycle=P_H2T_cycle, power_mismatch=power_mismatch,
            V_dot_in=V_dot_in, V_dot_out=V_dot_out,
        )
        return self

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    def to_csv_row(self):
        if self.results is None:
            raise RuntimeError("Call size() before to_csv_row().")
        return {f"expander__{k}": v for k, v in self.results.items()}

    # ------------------------------------------------------------------
    # Report
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

        c.print()
        c.rule("[bold white]GH2 PISTON EXPANDER SIZING[/bold white]")
        c.print()

        tbl_thermo = self._make_table(
            "GH2 Expander -- Thermodynamics  (ParaHydrogen, real gas)", [
            ("Inlet pressure (HC)",      f"{r['P_HC']:.0f}",       "bar"),
            ("Inlet temperature (HC)",   f"{r['T_HC']:.0f}",       "K"),
            ("Inlet density",            f"{r['rho_HC']:.3f}",     "kg/m³"),
            None,
            ("Outlet pressure (HD)",     f"{r['P_HD']:.1f}",       "bar"),
            ("Outlet temperature (HD)",  f"{r['T_HD']:.1f}",       "K"),
            ("Outlet density",           f"{r['rho_HD']:.4f}",     "kg/m³"),
            None,
            ("Pressure ratio",           f"{r['PR']:.1f}",            "-"),
            ("Density ratio (= vol ER)", f"{r['rho_ratio']:.1f}",     "-"),
            ("Isentropic efficiency",    f"{r['eta_H2T']:.2f}",       "-"),
            ("Specific work (actual)",   f"{r['w_spec_exp']/1e3:.1f}", "kJ/kg"),
            None,
            ("H2 mass flow",             f"{r['mdot']:.4f}",          "kg/s"),
            ("Expander power (calc)",    f"{r['P_exp_W']/1e3:.2f}",   "kW"),
            ("Expander power (cycle)",   f"{r['P_H2T_cycle']/1e3:.2f}","kW"),
        ], color="cyan")

        tbl_geom = self._make_table(
            "GH2 Expander -- Geometry  (Otto-cycle piston, single-acting)", [
            ("Crankshaft frequency",     f"{r['f_crank']:.0f}",       "Hz"),
            ("Number of cylinders",      f"{r['N_cyl']:.0f}",         "-"),
            None,
            ("Swept volume per cyl",     f"{r['V_swept']*1e6:.1f}",   "cc"),
            ("Bore",                     f"{r['bore']*1e3:.1f}",      "mm"),
            ("Stroke",                   f"{r['stroke']*1e3:.1f}",    "mm"),
            ("Wall thickness (Barlow)",  f"{r['t_wall']*1e3:.1f}",    "mm"),
            None,
            ("Mean piston speed",        f"{r['S_piston']:.2f}",      "m/s"),
            ("Mean effective pressure",  f"{r['Mean_eff_p']/1e5:.2f}","bar"),
            ("Peak cylinder pressure",   f"{r['P_peak']/1e5:.0f}",    "bar"),
            None,
            ("Implied clearance frac",   f"{r['c_clearance']:.3f}",   "-"),
            ("Vol flow at outlet",       f"{r['V_dot_out']*1e3:.3f}", "L/s"),
            ("Vol flow at inlet",        f"{r['V_dot_in']*1e6:.2f}",  "cc/s"),
        ], color="yellow")

        tbl_mass = self._make_table("GH2 Expander -- Mass Estimate", [
            ("Specific power",       f"{r['Sp_power']:.2f}",     "kW/kg"),
            ("Expander power",       f"{r['P_exp_W']/1e3:.2f}",  "kW"),
            ("Estimated mass",       f"{r['m_exp']:.1f}",        "kg"),
        ], color="magenta")

        c.print(Columns([tbl_thermo, tbl_geom], equal=False, expand=False))
        c.print()
        c.print(tbl_mass)
        c.print()

        # Warnings
        if r["S_piston"] > 12.0:
            c.print(
                f"[bold red]WARNING:[/bold red] Mean piston speed "
                f"{r['S_piston']:.1f} m/s exceeds 12 m/s. Consider more "
                f"cylinders or lower frequency."
            )
        if r["c_clearance"] < 0.03:
            c.print(
                f"[bold red]WARNING:[/bold red] Clearance fraction "
                f"{r['c_clearance']:.3f} < 0.03; consider two-stage expansion."
            )
        if r["c_clearance"] > 0.15:
            c.print(
                f"[bold yellow]NOTE:[/bold yellow] Clearance fraction "
                f"{r['c_clearance']:.3f} > 0.15; re-expansion losses non-trivial."
            )
        if r["power_mismatch"] > 0.01:
            c.print(
                f"[bold red]WARNING:[/bold red] Expander/cycle power mismatch "
                f"{r['power_mismatch']*100:.1f}%; check eta_H2T consistency."
            )


# ----------------------------------------------------------------------
# Standalone smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from config import Config

    cfg    = Config()
    engine = GasTurbineCycle.from_config(cfg).size()
    expander = PistonExpander.from_config(engine, cfg).size()
    expander.report()
