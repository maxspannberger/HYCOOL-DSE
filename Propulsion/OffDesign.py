import warnings
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.rule import Rule
from TurbineSizing import GasTurbineCycle
from plot_style import (
    apply_style, save_figure,
    cyan_tones, NEUTRAL_GREY, TEXT_GREY, ACCENT_WARN,
)




class OffDesignEvaluator:

    def __init__(
            self,
            engine,
            TIT_limit   = 1900,
            TIT_min     = 550,
            TIT_tol     = 1e-4,
            max_iter    = 100,
            Q_regen_max = None,
            T_pre_comp  = None,
            P_pre_comp  = None,
    ):
        self.engine     = engine
        self.TIT_limit  = TIT_limit
        self.TIT_min    = TIT_min
        self.TIT_tol    = TIT_tol
        self.max_iter   = max_iter
        self._console   = Console()

        results = engine.results
        d       = engine._design

        # Recuperator is sized at the design point — its maximum duty caps off-design heat transfer.
        self.Q_regen_W_max          = Q_regen_max if Q_regen_max is not None else results["Q_regen_W"]

        self.mdot_air_design        = results["mdot_f"] * results["ideal_OF"]
        self.Pc_design              = d["Pc"]                          # fixed compressor outlet pressure [bar]
        self.P2_design              = d["P2"]                          # fixed HPC exit pressure [bar]
        self.T1_design              = d["T1"]                          # fixed HPC inlet temperature [K]
        self.T2_design              = d["T2"]                          # fixed HPC exit temperature [K]
        self.Cp_HPC_design          = d["Cp_HPC"]
        self.P_HPC_design           = results["P_HPC_W"]

        self.w_compressor_design    = d["w_compressor"]                 # Compressor specific work
        self.dh_h2_design           = d["dh_h2"]                        # HEX heat addition per kg fuel
        self.w_H2T_design           = d["w_H2T"]                        # Gh2 turbine specific power output
        self.P3_H2_design           = d["P3_H2"]
        self.h3_actual_design       = d["h3_actual"]
        self.TH3_design             = d["TH3"]

        # --- Off-design H2 feed condition ---
        # Allow the cryogenic feed temperature and/or pressure into the H2 fuel
        # compressor to differ at off-design. h2_compressorout (state at H2 HEX
        # cold-side inlet) and w_compressor scale with these. If an override is
        # None, the design point value is used.
        self.T_pre_comp_od = (
            T_pre_comp if T_pre_comp is not None else engine.T_pre_comp
        )
        self.P_pre_comp_od = (
            P_pre_comp if P_pre_comp is not None else engine.P_pre_comp
        )
        feed_changed = (
            self.T_pre_comp_od != engine.T_pre_comp or
            self.P_pre_comp_od != engine.P_pre_comp
        )
        if feed_changed:
            h1_od = PropsSI("H", "P", self.P_pre_comp_od*1e5,
                            "T", self.T_pre_comp_od, engine.fluid)
            s1_od = PropsSI("S", "P", self.P_pre_comp_od*1e5,
                            "T", self.T_pre_comp_od, engine.fluid)
            h2s_od = PropsSI("H", "P", engine.PH1*1e5, "S", s1_od, engine.fluid)
            h2_od  = h1_od + (h2s_od - h1_od) / engine.eta_compressor
            self.h2_compressorout = h2_od
            self.w_compressor_od  = h2_od - h1_od
        else:
            self.h2_compressorout = d["h2_compressorout"]
            self.w_compressor_od  = d["w_compressor"]

    # ------------------------------------------------------------------
    # Build from external config
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, engine, cfg):
        """
        Construct from a sized `GasTurbineCycle` and a `Config` (or
        `OffDesignConfig`) defined in config.py.
        """
        o = cfg.offdesign if hasattr(cfg, "offdesign") else cfg
        return cls(
            engine      = engine,
            TIT_limit   = o.TIT_limit,
            TIT_min     = o.TIT_min,
            TIT_tol     = o.TIT_tol,
            max_iter    = o.max_iter,
            Q_regen_max = o.Q_regen_max,
            T_pre_comp  = o.T_pre_comp,
            P_pre_comp  = o.P_pre_comp,
        )

    # ------------------------------------------------------------------
    # Flat dict for CSV export -- given a single evaluate() result
    # ------------------------------------------------------------------
    @staticmethod
    def to_csv_row(result, prefix="offdesign"):
        """
        Flatten an evaluate() result dict into CSV-friendly columns.
        `prefix` is prepended to every key so multiple off-design points can
        coexist in one wide row (use e.g. "offdesign_OEI" vs "offdesign_cruise").
        """
        if result is None:
            return {}
        scalar_keys = [
            "P_shaft", "TIT_od", "mdot_f", "mdot_air", "mdot_aux",
            "aux_active", "aux_fraction",
            "OF", "T2p", "T5", "T_exh_final",
            "TH2", "T_hex_hot_in", "P_H2T_W", "P_H2comp_W", "h2_net_W",
            "Q_regen_W", "Q_regen_W_max", "regen_capped",
            "eta_total", "SFC", "SFC_hr", "q_in_W",
            "tit_exceeded", "hex_feasible", "approach_min",
        ]
        return {f"{prefix}__{k}": result[k] for k in scalar_keys if k in result}

    def cycle_at_TIT(self, TIT_od, mdot_air=None):
        """
        Solve the thermo cycle at the new TIT.

        mdot_air : optional override of the air mass flow through the engine.
                   Used by the auxiliary-intake mode, where extra ram air is
                   admitted so that more fuel can be burnt at the same TIT.
                   When None, the design air flow is used.
        """
        if mdot_air is None:
            mdot_air = self.mdot_air_design

        e = self.engine
        Pc = self.Pc_design

        # High pressure turbine analysis:
        PR_HPT_eff  = (Pc / e.P_ambient) * e.eta_CC_p * (e.eta_HEX**2) if e.FULL_EXPANSION else 7.0

        g_t         = e._air_gamma(Pc, TIT_od)
        T4          = TIT_od
        P4          = Pc * e.eta_CC_p
        T5s         = T4 * (1 / PR_HPT_eff) **((g_t - 1)/ g_t)
        P5          = Pc / PR_HPT_eff
        T5          = T4 - (T4 - T5s) * e.eta_HPT
        Cp_HPT      = e._air_cp(P5, 0.5 * (T4+T5))

        # Recuperator at new T5

        T2          = self.T2_design        # Fixed HPC exit temp (for now)
        cp_air_reg  = e._air_cp(self.P2_design, 0.5 * (T2 + T5))

        def OF_for(T_air_in):
            return (e.LHV_H2 * e.eta_CC / (e._air_h(Pc, TIT_od) - e._air_h(Pc, T_air_in)))

        # Convert the duty cap into the maximum air-side temperature rise it can support.
        # Q_regen_W = mdot_air * cp_air_reg * (T2p - T2)  =>  dT_pre_max = Q_max / (mdot_air * cp_air_reg)
        # Note: with aux air engaged, the same fixed recuperator must heat a larger air
        # stream, so the achievable preheat per kg drops -- this is captured by mdot_air
        # being the actual (possibly augmented) flow rather than the design value.
        dT_pre_max = self.Q_regen_W_max / (mdot_air * cp_air_reg)

        # --- Determine the H2 HEX hot-side inlet temperature ---
        # With REGEN_FIRST=True the air-side preheat dpre and resulting OF do not
        # depend on dh_h2 (the recuperator sees T5 directly), so T_after_reg and
        # hence T_hex_hot_in are fully determined before the O/F iteration. With
        # REGEN_FIRST=False or no recuperator, T_hex_hot_in = T5 immediately.
        if e.USE_REGEN and e.REGEN_FIRST:
            dpre_ideal   = e.eta_regen * (T5 - T2)
            dpre         = min(dpre_ideal, dT_pre_max)
            regen_capped = dpre_ideal > dT_pre_max
            T2p          = T2 + dpre
            OF           = OF_for(T2p)
            r_pre        = OF * cp_air_reg / ((OF + 1) * Cp_HPT)
            T_hex_hot_in = T5 - dpre * r_pre
        else:
            T_hex_hot_in = T5
            regen_capped = False
            # OF/T2p resolved below for these cases.
            OF  = OF_for(T2)
            T2p = T2

        # --- Set TH2 to its maximum (= H2 HEX hot-side inlet) and derive
        # downstream H2-side quantities per kg fuel ---
        TH2_od      = T_hex_hot_in
        h_hexout    = PropsSI("H", "P", e.PH1 * 1e5, "T", TH2_od, e.fluid)
        dh_h2       = h_hexout - self.h2_compressorout
        s2_in_h2    = PropsSI("S", "P", e.PH1 * 1e5, "T", TH2_od, e.fluid)
        h3_ideal_h2 = PropsSI("H", "P", self.P3_H2_design * 1e5, "S", s2_in_h2, e.fluid)
        w_H2T       = (h_hexout - h3_ideal_h2) * e.eta_H2T

        # --- Finish exhaust/O-F solve ---
        if e.USE_REGEN and e.REGEN_FIRST:
            # OF, T2p, T_hex_hot_in already fixed; just compute final stack T.
            T_exh_final = T_hex_hot_in - dh_h2 / ((OF + 1) * Cp_HPT)
        elif e.USE_REGEN:
            # REGEN_FIRST=False: T_after_hex couples T2p -> OF -> dpre -> T2p.
            T_exh_final = T5
            for _ in range(50):
                r            = OF * cp_air_reg / ((OF + 1) * Cp_HPT)
                T_after_hex  = T5 - dh_h2 / ((OF + 1) * Cp_HPT)
                T_reg_in     = T_after_hex
                dpre_ideal   = e.eta_regen * max(T_reg_in - T2, 0.0)
                dpre         = min(dpre_ideal, dT_pre_max)
                regen_capped = dpre_ideal > dT_pre_max
                T2p          = T2 + dpre
                T_exh_final  = T_reg_in - dpre * r
                OF_new       = OF_for(T2p)
                if abs(OF_new - OF) < 1e-9:
                    OF = OF_new
                    break
                OF = OF_new
        else:
            # No recuperator.
            T_exh_final = T5 - dh_h2 / ((OF + 1) * Cp_HPT)

        Q_regen_per_mf  = OF * cp_air_reg * (T2p - T2)

        return {
        "TIT_od":        TIT_od,
        "OF":            OF,
        "T2p":           T2p,
        "T4":            T4,   "P4": P4,
        "T5":            T5,   "P5": P5,
        "Cp_HPT":        Cp_HPT,
        "PR_HPT_eff":    PR_HPT_eff,
        "gamma":         g_t,
        "T_hex_hot_in":  T_hex_hot_in,
        "T_exh_final":   T_exh_final,
        "Q_regen_per_mf": Q_regen_per_mf,
        "cp_air_reg":    cp_air_reg,
        "regen_capped":  regen_capped,
        "TH2":           TH2_od,
        "dh_h2":         dh_h2,
        "w_H2T":         w_H2T,
    }


    def _net_power(self, TIT_od, mdot_aux=0.0):
        """
        Net shaft power at a given TIT and (optional) auxiliary air flow.

        mdot_aux : extra air admitted through an auxiliary intake (like the JT3D
                   blow-in doors). It is taken to be processed through the HPC at
                   the same pressure ratio and specific work as the design flow,
                   so HPC absorbed power scales linearly with the augmented flow.
        """
        mdot_air = self.mdot_air_design + mdot_aux
        od       = self.cycle_at_TIT(TIT_od, mdot_air=mdot_air)
        mdot_f   = mdot_air / od["OF"]
        mdot_tot = mdot_air + mdot_f

        P_HPT  = od["Cp_HPT"] * mdot_tot * (od["T4"] - od["T5"])
        P_HPC  = self.P_HPC_design * (mdot_air / self.mdot_air_design)
        P_H2T  = od["w_H2T"]          * mdot_f
        P_comp = self.w_compressor_od * mdot_f

        return (P_HPT + P_H2T - P_HPC - P_comp) * self.engine.eta_mech
        
    # Same as for TurbineSizing.py
    def _mdot_f_for_power(self, P_target_od, TIT_od, od):
        """
        Secant solve for the fuel flow that delivers P_target_od at TIT_od.

        Returns converged mdot_f [kg/s].
        """
        # Seed: scale design-point fuel flow by power ratio as initial guess
        des  = self.engine.results
        x0   = des["mdot_f"] * (P_target_od / des["total_net_W"]) * 0.9
        x1   = x0 * 1.1

        f0 = self._net_power(TIT_od, x0, od) - P_target_od
        f1 = self._net_power(TIT_od, x1, od) - P_target_od

        for _ in range(50):
            if abs(f1) < 1.0 or f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if x2 <= 0:
                x2 = 0.5 * x1
            f0, x0 = f1, x1
            f1 = self._net_power(TIT_od, x2, od) - P_target_od
            x1 = x2

        return x1
    
    def evaluate(self, P_shaft):
        from scipy.optimize import brentq

        P_at_TIT_limit = self._net_power(self.TIT_limit, mdot_aux=0.0)

        aux_active = False
        mdot_aux   = 0.0

        if P_shaft <= P_at_TIT_limit:
            # Normal mode: TIT is free, mdot_aux = 0
            P_lo = self._net_power(self.TIT_min, mdot_aux=0.0)
            if not (P_lo <= P_shaft <= P_at_TIT_limit):
                raise ValueError(
                    f"Could not bracket a solution for P_shaft={P_shaft/1e6:.3f} MW. "
                    f"No-aux feasible range is [{P_lo/1e6:.3f}, {P_at_TIT_limit/1e6:.3f}] MW "
                    f"for TIT in [{self.TIT_min}, {self.TIT_limit}] K."
                )
            TIT_od = brentq(
                lambda T: self._net_power(T, mdot_aux=0.0) - P_shaft,
                self.TIT_min, self.TIT_limit, xtol=0.1,
            )
        else:
            # Aux mode: TIT pinned at limit, solve for the extra ram-air flow.
            aux_active = True
            TIT_od     = self.TIT_limit

            # Grow the upper bracket until it exceeds P_shaft (doubling each step).
            # Cap the search at 5x the design air flow as a sanity limit.
            mdot_aux_hi = self.mdot_air_design
            for _ in range(8):
                if self._net_power(TIT_od, mdot_aux=mdot_aux_hi) >= P_shaft:
                    break
                mdot_aux_hi *= 2.0
            else:
                raise ValueError(
                    f"P_shaft={P_shaft/1e6:.3f} MW unreachable even with "
                    f"{mdot_aux_hi/self.mdot_air_design:.1f}x extra air at "
                    f"TIT={TIT_od:.0f} K. Engine cannot deliver this power."
                )

            mdot_aux = brentq(
                lambda m: self._net_power(TIT_od, mdot_aux=m) - P_shaft,
                0.0, mdot_aux_hi, xtol=1e-4,
            )

        # Recover all quantities at the converged operating point
        mdot_air = self.mdot_air_design + mdot_aux
        od       = self.cycle_at_TIT(TIT_od, mdot_air=mdot_air)
        mdot_f   = mdot_air / od["OF"]
        mdot_tot = mdot_air + mdot_f

        e         = self.engine
        q_in      = mdot_f * e.LHV_H2
        eta_total = P_shaft / q_in
        SFC       = mdot_f / P_shaft
        SFC_hr    = SFC * 3.6e6

        # HEX pinch check
        C_hot     = mdot_tot * od["Cp_HPT"]
        Q_hex     = od["dh_h2"] * mdot_f
        T_hot_in  = od["T_hex_hot_in"]
        T_hot_out = T_hot_in - Q_hex / C_hot

        q_arr    = np.linspace(0, Q_hex, 100)
        h_cold   = self.h2_compressorout + q_arr / mdot_f
        T_cold   = np.array([PropsSI("T", "P", e.PH1*1e5, "H", h, e.fluid) for h in h_cold])
        T_hot_arr = T_hot_out + q_arr / C_hot
        approach_min = (T_hot_arr - T_cold).min()

        # Afterburner heat addition

        

        return {
            "P_shaft":        P_shaft,
            "TIT_od":         TIT_od,
            "mdot_f":         mdot_f,
            "mdot_air_check": mdot_f * od["OF"],
            "OF":             od["OF"],
            "T2p":            od["T2p"],
            "T5":             od["T5"],
            "T_exh_final":    od["T_exh_final"],
            "TH2":            od["TH2"],
            "T_hex_hot_in":   od["T_hex_hot_in"],
            "P_H2T_W":        od["w_H2T"]          * mdot_f,
            "P_H2comp_W":     self.w_compressor_od * mdot_f,
            "h2_net_W":       (od["w_H2T"] - self.w_compressor_od) * mdot_f,
            "Q_regen_W":      od["Q_regen_per_mf"] * mdot_f,
            "Q_regen_W_max":  self.Q_regen_W_max,
            "regen_capped":   od["regen_capped"],
            "eta_total":      eta_total,
            "SFC":            SFC,
            "SFC_hr":         SFC_hr,
            "q_in_W":         q_in,
            "tit_exceeded":   TIT_od > self.TIT_limit + 1e-6,
            "aux_active":     aux_active,
            "mdot_aux":       mdot_aux,
            "mdot_air":       mdot_air,
            "aux_fraction":   mdot_aux / self.mdot_air_design,
            "hex_feasible":   approach_min > 0,
            "approach_min":   approach_min,
        }
    
    def sweep(self, P_min, P_max, n_points=30):
        """
        Evaluate performance across a range of shaft power levels.

        """
        powers  = np.linspace(P_min, P_max, n_points)
        results = []

        for P in powers:
            try:
                results.append(self.evaluate(P))
            except ValueError as exc:
                warnings.warn(f"Skipping P={P/1e6:.3f} MW: {exc}", RuntimeWarning)

        return results
    
    def report(self, result):

        c   = self._console
        r   = result
        des = self.engine.results

        tit_color = "red bold" if r["tit_exceeded"] else ("yellow bold" if r["aux_active"] else "green bold")
        if r["tit_exceeded"]:
            tit_warn = "  [red][!] EXCEEDS TIT LIMIT[/red]"
        elif r["aux_active"]:
            tit_warn = "  [yellow][!] PINNED AT LIMIT (AUX OPEN)[/yellow]"
        else:
            tit_warn = ""
        hex_color = "green" if r["hex_feasible"] else "red"
        hex_warn  = "" if r["hex_feasible"] else "  [red][!] PINCH VIOLATED[/red]"

        c.print(Rule(
            f"[bold cyan]Off-Design Point: "
            f"{r['P_shaft']/1e6:.3f} MW[/bold cyan]"
        ))

        t = Table(box=box.ROUNDED, header_style="bold magenta", show_lines=True)
        t.add_column("Parameter",        style="cyan",  justify="left",  min_width=36)
        t.add_column("Off-design",       style="yellow", justify="right", min_width=14)
        t.add_column("Design (cruise)",  style="dim",   justify="right", min_width=14)
        t.add_column("Units",            style="dim",   justify="left")

        t.add_row(
            "Net shaft power",
            f"{r['P_shaft']/1e6:.3f}",
            f"{des['total_net_W']/1e6:.3f}",
            "MW",
        )
        t.add_row(
            f"TIT{tit_warn}",
            f"[{tit_color}]{r['TIT_od']:.1f}[/{tit_color}]",
            f"{des['TIT']:.1f}",
            "K",
        )
        t.add_row(
            "Fuel mass flow",
            f"{r['mdot_f']*1e3:.3f}",
            f"{des['mdot_f']*1e3:.3f}",
            "g/s",
        )
        t.add_row(
            "O/F ratio",
            f"{r['OF']:.2f}",
            f"{des['ideal_OF']:.2f}",
            "-",
        )
        t.add_row(
            "Combustor air inlet T (T2')",
            f"{r['T2p']:.1f}",
            f"{des['T2p']:.1f}",
            "K",
        )
        t.add_row(
            "HPT exit T (T5)",
            f"{r['T5']:.1f}",
            f"{des['T5']:.1f}",
            "K",
        )
        t.add_row(
            "Final exhaust T",
            f"{r['T_exh_final']:.1f}",
            f"{des['T_exh_final']:.1f}",
            "K",
        )
        t.add_row(
            "H2 HEX outlet T (TH2, dynamic)",
            f"{r['TH2']:.1f}",
            f"{des['TH2']:.1f}",
            "K",
        )
        t.add_row(
            "GH2 expander output",
            f"{r['P_H2T_W']/1e3:.2f}",
            f"{des['P_H2T_W']/1e3:.2f}",
            "kW",
        )
        t.add_row(
            "H2 compressor draw",
            f"{r['P_H2comp_W']/1e3:.2f}",
            f"{des['Power_compressor_W']/1e3:.2f}",
            "kW",
        )
        h2_net_color    = "green" if r["h2_net_W"] >= 0 else "red"
        h2_net_color_des = "green" if des["h2_net_W"] >= 0 else "red"
        t.add_row(
            "Net GH2 circuit",
            f"[{h2_net_color}]{r['h2_net_W']/1e3:+.2f}[/{h2_net_color}]",
            f"[{h2_net_color_des}]{des['h2_net_W']/1e3:+.2f}[/{h2_net_color_des}]",
            "kW",
        )
        regen_color = "yellow bold" if r["regen_capped"] else "yellow"
        regen_warn  = "  [yellow][!] CAPPED[/yellow]" if r["regen_capped"] else ""
        t.add_row(
            f"Recuperator duty{regen_warn}",
            f"[{regen_color}]{r['Q_regen_W']/1e6:.3f}[/{regen_color}]",
            f"{des['Q_regen_W']/1e6:.3f}",
            "MW",
        )
        t.add_row(
            "Thermal efficiency",
            f"{r['eta_total']*100:.1f}",
            f"{des['eta_total']*100:.1f}",
            "%",
        )
        t.add_row(
            "SFC",
            f"{r['SFC_hr']:.4f}",
            f"{des['mdot_f']/des['total_net_W']*3.6e6:.4f}",
            "kg/kW/hr",
        )
        t.add_row(
            f"HEX min approach T{hex_warn}",
            f"[{hex_color}]{r['approach_min']:.1f}[/{hex_color}]",
            "-",
            "K",
        )
        aux_color = "yellow bold" if r["aux_active"] else "dim"
        t.add_row(
            "Auxiliary air intake flow",
            f"[{aux_color}]{r['mdot_aux']:.4f}[/{aux_color}]",
            "0.0000",
            "kg/s",
        )
        t.add_row(
            "Auxiliary air fraction",
            f"[{aux_color}]{r['aux_fraction']*100:.1f}[/{aux_color}]",
            "0.0",
            "%",
        )
        t.add_row(
            "Total engine air flow",
            f"{r['mdot_air']:.4f}",
            f"{self.mdot_air_design:.4f}",
            "kg/s",
        )
        t.add_row(
            "Air flow check (mdot_f * OF)",
            f"{r['mdot_air_check']:.4f}",
            f"{self.mdot_air_design:.4f}",
            "kg/s",
        )
        c.print(t)

        if r["aux_active"]:
            c.print(Panel(
                f"[yellow bold]AUX INTAKE ENGAGED:[/yellow bold] without auxiliary air "
                f"the demanded P_shaft = {r['P_shaft']/1e6:.3f} MW would force "
                f"TIT above the {self.TIT_limit:.0f} K limit. The auxiliary intake "
                f"(JT3D blow-in-door style) admits an extra "
                f"[bold]{r['mdot_aux']:.3f} kg/s[/bold] of ram air "
                f"([bold]+{r['aux_fraction']*100:.1f}%[/bold] over the design "
                f"{self.mdot_air_design:.3f} kg/s), pinning TIT exactly at the "
                f"limit while the additional fuel keeps O/F constant and produces "
                f"the required shaft power.\n"
                f"Cost: HPC absorbed power scales with total air flow, and the "
                f"fixed-size recuperator's per-kg preheat drops, both of which "
                f"erode thermal efficiency at this operating point.",
                border_style="yellow",
                title="Auxiliary Air Intake Active",
            ))
        elif r["tit_exceeded"]:
            c.print(Panel(
                f"[red bold]WARNING:[/red bold] TIT = {r['TIT_od']:.1f} K still "
                f"exceeds the material limit of {self.TIT_limit:.1f} K even with "
                f"the auxiliary intake fully open. This operating point is not "
                f"sustainable; consider derating power or revisiting the engine "
                f"sizing.",
                border_style="red",
                title="TIT Limit Exceeded",
            ))

        if not r["hex_feasible"]:
            c.print(Panel(
                f"[red bold]WARNING:[/red bold] H2 HEX pinch violated at this "
                f"operating point (approach_min = {r['approach_min']:.1f} K < 0).\n"
                f"The heat exchanger cannot transfer the required duty without a "
                f"temperature cross. Review HEX sizing or H2 inlet conditions.",
                border_style="red",
                title="HEX Pinch Violated",
            ))

    # ------------------------------------------------------------------
    # Public: plot sweep results
    # ------------------------------------------------------------------
    def plot_sweep(self, sweep_results, save_dir=None, show=True):
        """
        Four-panel off-design sweep: TIT, thermal efficiency, SFC, O/F.

        All curves share one muted-cyan tone (per-panel tone selected from
        the palette) so the figure reads as one cohesive figure rather than
        four candy-coloured plots.
        """
        if not sweep_results:
            raise ValueError("sweep_results is empty -- nothing to plot.")

        apply_style()

        des = self.engine.results

        P_MW  = np.array([r["P_shaft"]   / 1e6 for r in sweep_results])
        TIT   = np.array([r["TIT_od"]          for r in sweep_results])
        eta   = np.array([r["eta_total"] * 100 for r in sweep_results])
        SFC   = np.array([r["SFC_hr"]          for r in sweep_results])
        OF    = np.array([r["OF"]              for r in sweep_results])

        P_design_MW = des["total_net_W"] / 1e6

        # Four cyan tones, one per panel -- different enough to feel distinct
        # when the four subplots sit side by side, all in the same family.
        c_tit, c_eta, c_sfc, c_of = cyan_tones(4, lightest=2, darkest=6)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
        fig.suptitle(
            "Off-design performance sweep   "
            "(fixed air mass flow, variable TIT and fuel flow)",
            fontsize=12.5, color=TEXT_GREY, y=0.995,
        )

        def _decorate(ax, ylabel, title, curve_colour, show_tit_limit=False):
            ax.axvline(P_design_MW, color=NEUTRAL_GREY, linestyle=':', lw=1.0,
                       label=f"Design point ({P_design_MW:.2f} MW)")
            if show_tit_limit:
                ax.axhline(self.TIT_limit, color=ACCENT_WARN, linestyle='--', lw=1.2,
                           label=f"TIT limit ({self.TIT_limit:.0f} K)")
            ax.set_xlabel("Net shaft power  [MW]")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="best", fontsize=8.5)

        # TIT vs power
        ax = axes[0, 0]
        ax.plot(P_MW, TIT, '-o', color=c_tit, lw=1.8, ms=4.2)
        _decorate(ax, "TIT  [K]", "Turbine inlet temperature", c_tit,
                  show_tit_limit=True)

        # Thermal efficiency vs power
        ax = axes[0, 1]
        ax.plot(P_MW, eta, '-o', color=c_eta, lw=1.8, ms=4.2)
        _decorate(ax, "Thermal efficiency  [%]", "Thermal efficiency", c_eta)

        # SFC vs power
        ax = axes[1, 0]
        ax.plot(P_MW, SFC, '-o', color=c_sfc, lw=1.8, ms=4.2)
        _decorate(ax, "SFC  [kg/kW/hr]", "Specific fuel consumption", c_sfc)

        # O/F vs power
        ax = axes[1, 1]
        ax.plot(P_MW, OF, '-o', color=c_of, lw=1.8, ms=4.2)
        _decorate(ax, "O/F  [–]", "Oxidiser-to-fuel ratio", c_of)

        fig.tight_layout(rect=(0, 0, 1, 0.97))

        if save_dir is not None:
            return save_figure(fig, "offdesign_sweep", save_dir)
        if show:
            plt.show()
        return fig

if __name__ == "__main__":
    # Standalone smoke test: drive everything from config.py.
    from config import Config

    cfg = Config()
    engine = GasTurbineCycle.from_config(cfg).size()
    engine.report()

    evaluator = OffDesignEvaluator.from_config(engine, cfg)

    for P in cfg.offdesign.P_shaft_cases:
        print(f"\n--- Off-design point at {P/1e6:.3f} MW ---")
        evaluator.report(evaluator.evaluate(P))

    print("\n--- Power sweep ---")
    sweep = evaluator.sweep(
        P_min=cfg.offdesign.P_sweep_min,
        P_max=cfg.offdesign.P_sweep_max,
        n_points=cfg.offdesign.P_sweep_n,
    )
    evaluator.plot_sweep(sweep)