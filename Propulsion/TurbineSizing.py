"""
TurbineSizing.py
====================
Hydrogen-fuelled recuperated gas turbine cycle -- design-point sizing.

Cycle architecture
------------------
Inlet -> RAM diffuser -> HPC -> [recuperator] -> combustor -> HPT -> exhaust
                                                                        |
                                                 [H2 HEX] <------------+
                                                     |
                                            GH2 expander turbine -> combustor injector

The main gas path (air side) and the hydrogen fuel circuit are modelled
separately, then coupled through:
  (a) the H2 HEX, which draws heat from the exhaust stream, and
  (b) the combustor, where the conditioned GH2 is injected.

Station definitions (air / gas path)
-------------------------------------
  1   Pre-HPC          (after ram diffuser)
  2   Post-HPC         (before recuperator)
  2'  Post-recuperator (combustor air inlet)
  3   Combustor inlet  (= cycle high pressure, Pc)
  4   Post-combustor   (= Pc * eta_CC_p, slight pressure drop across CC)
  5   Post-HPT         (near-ambient when FULL_EXPANSION = True)

Hydrogen circuit stations
--------------------------
  HA  Cryogenic feed   (P_pre_comp, T_pre_comp)
  HB  Post-H2 compressor  (PH1)
  HC  Post-HEX / turbine inlet  (PH1, TH2)
  HD  Post-H2 turbine  (P3_H2 = Pc * 1.1, open path into combustor)

Key design choices / flags
---------------------------
  FULL_EXPANSION  -- expand HPT all the way to ambient pressure rather than a
                     fixed PR_HPT; leaves no unused pressure energy in exhaust.
  USE_REGEN       -- enable the recuperator.
  REGEN_FIRST     -- controls exhaust routing order:
                       False (default): exhaust -> H2 HEX -> recuperator
                       True           : exhaust -> recuperator -> H2 HEX
"""

from rocketcea.cea_obj_w_units import CEA_Obj
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.rule import Rule


class GasTurbineCycle:
    """
    Design-point model of a hydrogen-fuelled recuperated gas turbine.

    All cycle parameters are set in the constructor with sensible defaults
    matching the original sizing target. Call `size()` to run the full solve
    and populate `self.results`. Reporting and plotting methods are available
    once `size()` has been called.

    Parameters
    ----------
    P_target : float
        Net shaft power target [W].
    P_ambient : float
        Ambient static pressure [bar] -- set for cruise altitude.
    M0 : float
        Cruise Mach number.
    TIT : float
        Turbine inlet temperature target [K].
    PR_HPC : float
        High-pressure compressor pressure ratio.
    eta_HPC : float
        Isentropic efficiency of the HPC.
    eta_CC : float
        Combustor thermal efficiency (fuel energy release fraction).
    eta_CC_p : float
        Combustor pressure recovery (P4 / P3).
    eta_HPT : float
        Isentropic efficiency of the HPT.
    eta_HEX : float
        Heat exchanger effectiveness (used in full-expansion PR calculation).
    eta_mech : float
        Mechanical transmission efficiency (shaft to output).
    eta_diff : float
        Ram diffuser pressure recovery.
    eta_regen : float
        Recuperator thermal effectiveness.
    eta_regen_p : float
        Recuperator pressure recovery on the air side.
    USE_REGEN : bool
        Enable the recuperator.
    FULL_EXPANSION : bool
        Expand HPT to ambient pressure rather than a fixed PR.
    REGEN_FIRST : bool
        Exhaust routing: True = recuperator before H2 HEX, False = H2 HEX first.
    Regen_Fraction : float
        Fraction of exhaust routed through the recuperator (reserved for
        partial-flow recuperation studies; currently unused in the energy balance).
    P_pre_comp : float
        H2 feed pressure before the fuel compressor [bar].
    T_pre_comp : float
        H2 feed temperature before the fuel compressor [K].
    PH1 : float
        H2 pressure after the fuel compressor / through the HEX [bar].
    TH2 : float
        Target H2 temperature at HEX outlet / turbine inlet [K].
    eta_compressor : float
        Isentropic efficiency of the H2 fuel compressor.
    eta_H2T : float
        Isentropic efficiency of the GH2 expander turbine.
    fluid : str
        CoolProp fluid name for the hydrogen working fluid.
    LHV_H2 : float
        Lower heating value of hydrogen [J/kg].
    mdot_f_init : float
        Initial fuel-flow guess for the sizing solver [kg/s].
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        # Sizing target
        P_target    = 2.85e6,
        # Ambient / flight conditions
        P_ambient   = 0.38,
        M0          = 0.7,
        # Core cycle
        TIT         = 1500.0,
        PR_HPC      = 7.0,
        eta_HPC     = 0.90,
        eta_CC      = 0.995,
        eta_CC_p    = 0.99,
        eta_HPT     = 0.90,
        eta_HEX     = 0.90,
        eta_mech    = 0.99,
        eta_diff    = 0.97,
        # Recuperator
        eta_regen   = 0.775,
        eta_regen_p = 0.95,
        USE_REGEN   = True,
        FULL_EXPANSION = True,
        REGEN_FIRST    = True,
        Regen_Fraction = 0.5,
        # Hydrogen circuit
        P_pre_comp  = 10.0,
        T_pre_comp  = 60.0,
        PH1         = 150.0,
        TH2         = 800.0,
        eta_compressor = 0.70,
        eta_H2T     = 0.90,
        fluid       = "ParaHydrogen",
        LHV_H2      = 120e6,
        # Solver
        mdot_f_init = 0.155,
    ):
        # --- Store all parameters as instance attributes ---
        self.P_target       = P_target
        self.P_ambient      = P_ambient
        self.M0             = M0
        self.TIT            = TIT
        self.PR_HPC         = PR_HPC
        self.eta_HPC        = eta_HPC
        self.eta_CC         = eta_CC
        self.eta_CC_p       = eta_CC_p
        self.eta_HPT        = eta_HPT
        self.eta_HEX        = eta_HEX
        self.eta_mech       = eta_mech
        self.eta_diff       = eta_diff
        self.eta_regen      = eta_regen
        self.eta_regen_p    = eta_regen_p
        self.USE_REGEN      = USE_REGEN
        self.FULL_EXPANSION = FULL_EXPANSION
        self.REGEN_FIRST    = REGEN_FIRST
        self.Regen_Fraction = Regen_Fraction
        self.P_pre_comp     = P_pre_comp
        self.T_pre_comp     = T_pre_comp
        self.PH1            = PH1
        self.TH2            = TH2
        self.eta_compressor = eta_compressor
        self.eta_H2T        = eta_H2T
        self.fluid          = fluid
        self.LHV_H2         = LHV_H2
        self.mdot_f_init    = mdot_f_init

        # Results populated by size()
        self.results  = None
        self.history  = None
        self._design  = None   # intermediate design-point dict from _solve_design()

        self._console = Console()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def size(self, cea=None):
        """
        Run the full design-point solve and fuel-flow sizing.

        Optionally accepts a RocketCEA CEA_Obj instance for combustion product
        molecular weight (used only for the specific gas constant Rs in the
        report). If None, Rs is omitted from the output.

        Populates self.results and self.history. Returns self for chaining.
        """
        self._design          = self._solve_design(cea)
        mdot_f, res, history  = self._size_fuel_flow()
        self.results          = res
        self.history          = history
        return self

    # ------------------------------------------------------------------
    # Air property helpers (thin wrappers around CoolProp)
    # ------------------------------------------------------------------
    @staticmethod
    def _air_gamma(p_bar, T):
        """Ratio of specific heats for air at (p_bar, T)."""
        return (PropsSI('CPMASS', 'P', p_bar*1e5, 'T', T, 'Air') /
                PropsSI('CVMASS', 'P', p_bar*1e5, 'T', T, 'Air'))

    @staticmethod
    def _air_cp(p_bar, T):
        """Isobaric specific heat of air [J/kg/K] at (p_bar, T)."""
        return PropsSI('CPMASS', 'P', p_bar*1e5, 'T', T, 'Air')

    @staticmethod
    def _air_h(p_bar, T):
        """Specific enthalpy of air [J/kg] at (p_bar, T)."""
        return PropsSI('H', 'P', p_bar*1e5, 'T', T, 'Air')

    # ------------------------------------------------------------------
    # Design-point solve (flow-independent quantities)
    # ------------------------------------------------------------------
    def _solve_design(self, cea=None):
        """
        Solve all thermodynamic quantities that are independent of fuel mass flow.

        Because the cycle is linear in mdot_f, separating the flow-independent
        part allows the fuel-flow sizing loop to converge in a single secant step.

        The recuperator couples the combustor air-inlet temperature (T2') to the
        oxidiser-to-fuel ratio (O/F): higher preheat -> leaner mixture -> different
        exhaust temperature -> different recuperator hot-side inlet. This coupling
        is resolved by a fixed-point iteration on O/F, which typically converges
        in < 10 steps.

        Returns a dict of cycle quantities normalised per unit or per kg of fuel
        where appropriate, ready for scaling by mdot_f in _run_cycle().
        """
        # --- Inlet / ram conditions ---
        # Stagnation temperature and pressure at HPC face after ram diffuser.
        T0 = 239.0   # static ambient temperature [K] at 25,000 ft
        P0 = self.P_ambient

        g_0 = self._air_gamma(P0, T0)
        T1  = T0 * (1 + ((g_0 - 1) / 2) * self.M0**2)
        P1  = P0 * (1 + ((g_0 - 1) / 2) * self.M0**2)**(g_0 / (g_0 - 1)) * self.eta_diff

        # --- High-pressure compressor (HPC) ---
        # Isentropic compression from P1 to P2, corrected for polytropic efficiency.
        g_c  = self._air_gamma(P1, T1)
        T2s  = T1 * self.PR_HPC**((g_c - 1) / g_c)      # ideal (isentropic) exit T
        P2   = P1 * self.PR_HPC
        T2   = T1 + (T2s - T1) / self.eta_HPC           # actual exit T
        Cp_HPC = self._air_cp(P2, 0.5*(T1 + T2))        # mean Cp across HPC

        # --- Cycle high pressure (Pc) ---
        # Pc is the air pressure entering the combustor after the recuperator
        # pressure drop. This is also the reference pressure for the HPT inlet.
        P2p = P2 * self.eta_regen_p if self.USE_REGEN else P2
        Pc  = P2p

        # --- HPT expansion ratio ---
        # FULL_EXPANSION: expand to ambient, accounting for HEX pressure losses.
        # Fixed PR: use the user-specified PR_HPT (kept as a fallback).
        PR_HPT_eff = (
            (Pc / self.P_ambient) * self.eta_CC_p * (self.eta_HEX**2)
            if self.FULL_EXPANSION else 7.0   # fallback fixed PR
        )

        # --- HPT inlet and exit conditions ---
        g_t  = self._air_gamma(Pc, self.TIT)
        T4   = self.TIT
        P4   = Pc * self.eta_CC_p                           # combustor pressure drop
        T5s  = T4 * (1 / PR_HPT_eff)**((g_t - 1) / g_t)     # isentropic exit T
        P5   = Pc / PR_HPT_eff
        T5   = T4 - (T4 - T5s) * self.eta_HPT               # actual exit T
        Cp_HPT = self._air_cp(P5, 0.5*(T4 + T5))

        # --- Hydrogen fuel circuit (per kg of fuel, flow-independent) ---
        # The H2 circuit is an open path: cryogenic feed -> compressor -> HEX
        # -> expander turbine -> combustor injector.
        h1  = PropsSI("H", "P", self.P_pre_comp*1e5, "T", self.T_pre_comp, self.fluid)
        s1  = PropsSI("S", "P", self.P_pre_comp*1e5, "T", self.T_pre_comp, self.fluid)
        h2s = PropsSI("H", "P", self.PH1*1e5, "S", s1, self.fluid)
        h2  = h1 + (h2s - h1) / self.eta_compressor         # actual compressor outlet enthalpy
        w_compressor = h2 - h1                              # compressor specific work [J/kg_fuel]
        h_hexout     = PropsSI("H", "P", self.PH1*1e5, "T", self.TH2, self.fluid)
        dh_h2        = h_hexout - h2                        # HEX heat addition [J/kg_fuel]

        # Mean air Cp across the recuperator (used in the fixed-point iteration below)
        cp_air_reg = self._air_cp(P2, 0.5*(T2 + T5))

        # --- Fixed-point iteration: O/F <-> combustor air-inlet temperature ---
        # The recuperator preheats the air from T2 to T2', which changes the
        # required O/F (leaner air -> lower fuel fraction). But T2' depends on
        # the exhaust temperature, which depends on O/F through the mass flow
        # ratio. Iterate to self-consistency.
        def OF_for(T_air_in):
            """O/F from combustor energy balance given preheated air inlet T."""
            return self.LHV_H2 * self.eta_CC / (
                self._air_h(Pc, self.TIT) - self._air_h(Pc, T_air_in)
            )

        OF            = OF_for(T2)
        T2p           = T2
        T_hex_hot_in  = T5
        T_exh_final   = T5

        for _ in range(50):
            if self.USE_REGEN:
                # Heat capacity ratio: air side / exhaust side
                r = OF * cp_air_reg / ((OF + 1) * Cp_HPT)

                if self.REGEN_FIRST:
                    # Exhaust -> recuperator -> H2 HEX
                    T_reg_in    = T5
                    T2p         = T2 + self.eta_regen * (T_reg_in - T2)
                    T_after_reg = T_reg_in - self.eta_regen * (T_reg_in - T2) * r
                    T_hex_hot_in = T_after_reg
                    T_exh_final  = T_hex_hot_in - dh_h2 / ((OF + 1) * Cp_HPT)
                else:
                    # Exhaust -> H2 HEX -> recuperator (default)
                    T_hex_hot_in = T5
                    T_after_hex  = T5 - dh_h2 / ((OF + 1) * Cp_HPT)
                    T_reg_in     = T_after_hex
                    dpre         = self.eta_regen * max(T_reg_in - T2, 0.0)
                    T2p          = T2 + dpre
                    T_exh_final  = T_reg_in - dpre * r
            else:
                # No recuperator: air enters combustor at T2, exhaust cools only via H2 HEX
                T2p          = T2
                T_hex_hot_in = T5
                T_exh_final  = T5 - dh_h2 / (OF_for(T2) + 1)

            OF_new = OF_for(T2p)
            if abs(OF_new - OF) < 1e-9:
                OF = OF_new
                break
            OF = OF_new

        if not self.USE_REGEN:
            # Recompute with converged O/F (the loop above used a placeholder)
            T_exh_final = T5 - dh_h2 / ((OF + 1) * Cp_HPT)

        # Recuperator duty per kg of fuel [J/kg_fuel]
        Q_regen_per_mf = OF * cp_air_reg * (T2p - T2)

        # --- GH2 expander turbine ---
        # The fuel is expanded from PH1 (post-HEX) down to slightly above Pc
        # (10% margin for injector pressure drop) before entering the combustor.
        P3_H2  = Pc * 1.1
        h2_in  = PropsSI('H', 'P', self.PH1*1e5, 'T', self.TH2, self.fluid)
        s2_in  = PropsSI('S', 'P', self.PH1*1e5, 'T', self.TH2, self.fluid)
        h3_ideal = PropsSI('H', 'P', P3_H2*1e5, 'S', s2_in, self.fluid)
        w_H2T    = (h2_in - h3_ideal) * self.eta_H2T   # specific work [J/kg_fuel]
        h3_actual = h2_in - w_H2T                       # actual turbine exit enthalpy
        TH3      = PropsSI('T', 'P', P3_H2*1e5, 'H', h3_actual, self.fluid)
        Cp_H2T   = (h2_in - h3_actual) / (self.TH2 - TH3) if self.TH2 != TH3 else 0.0

        # --- CEA combustion products: specific gas constant (reporting only) ---
        Rs = None
        if cea is not None:
            molwt, _ = cea.get_exit_MolWt_gamma(Pc=Pc, MR=OF, eps=1, frozen=1)
            Rs = 8.314 / (molwt / 1000)

        return {
            # Station conditions (air path)
            "T1": T1, "P1": P1,
            "T2": T2, "P2": P2,
            "T2p": T2p,
            "T4": T4, "P4": P4,
            "T5": T5, "P5": P5,
            "Pc": Pc,                   # true cycle high pressure (= P3 in station table)
            # Component Cp values
            "Cp_HPC": Cp_HPC, "Cp_HPT": Cp_HPT,
            "gamma": g_t, "PR_HPT_eff": PR_HPT_eff,
            # Combustion
            "OF": OF, "Rs": Rs, "TIT": self.TIT,
            # Recuperator
            "Q_regen_per_mf": Q_regen_per_mf, "cp_air_reg": cp_air_reg,
            "T_hex_hot_in": T_hex_hot_in, "T_exh_final": T_exh_final,
            # Hydrogen circuit (per kg fuel)
            "h2_compressorout": h2,
            "w_compressor": w_compressor,
            "dh_h2": dh_h2,
            "w_H2T": w_H2T,
            "TH3": TH3, "Cp_H2T": Cp_H2T,
            "P3_H2": P3_H2, "h3_actual": h3_actual,
        }

    # ------------------------------------------------------------------
    # Full cycle (scales design point by mdot_f)
    # ------------------------------------------------------------------
    def _run_cycle(self, mdot_f, d):
        """
        Scale the flow-independent design point by a given fuel mass flow.

        All power quantities [W] and mass flows [kg/s] are computed here.
        The HEX pinch-point sweep is also performed to check thermodynamic
        feasibility of the hydrogen heat exchanger.

        Parameters
        ----------
        mdot_f : float
            Fuel (H2) mass flow rate [kg/s].
        d : dict
            Design-point dict from _solve_design().

        Returns
        -------
        dict of all cycle results, suitable for reporting and sizing.
        """
        mdot_tot = mdot_f * (d["OF"] + 1)   # total mass flow: air + fuel [kg/s]

        # --- Component power contributions [W] ---
        P_HPC  = d["Cp_HPC"] * mdot_f * d["OF"] * (d["T2"]  - d["T1"])   # compressor demand
        P_HPT  = d["Cp_HPT"] * mdot_f * (d["OF"] + 1) * (d["T4"] - d["T5"])  # turbine output
        P_H2T  = d["w_H2T"]  * mdot_f       # GH2 expander output
        P_comp = d["w_compressor"] * mdot_f  # H2 fuel compressor demand

        # --- H2 HEX pinch-point sweep ---
        # Verifies that the hot exhaust is always hotter than the cold H2 at every
        # point along the HEX, i.e. no temperature cross (approach_min > 0 required).
        C_hot     = mdot_tot * d["Cp_HPT"]
        Q_tot     = d["dh_h2"] * mdot_f
        h_in      = d["h2_compressorout"]
        T_hot_in  = d["T_hex_hot_in"]
        T_hot_out = T_hot_in - Q_tot / C_hot

        N = 200
        q       = np.linspace(0, Q_tot, N)
        h_cold  = h_in + q / mdot_f
        T_cold  = np.array([
            PropsSI("T", "P", self.PH1*1e5, "H", h, self.fluid) for h in h_cold
        ])
        T_hot        = T_hot_out + q / C_hot
        approach     = T_hot - T_cold
        approach_min = approach.min()
        approach_loc = q[approach.argmin()] / Q_tot

        # --- Net power and efficiency ---
        gaspath_net = P_HPT  - P_HPC
        h2_net      = P_H2T  - P_comp
        total_net   = (P_HPT + P_H2T - P_HPC - P_comp) * self.eta_mech

        q_in        = mdot_f * self.LHV_H2
        eta_total   = total_net   / q_in
        eta_gaspath = gaspath_net / q_in

        return {
            # Mass flows
            "mdot_f": mdot_f, "mdot_tot": mdot_tot,
            # Combustion / gas properties
            "ideal_OF": d["OF"], "Rs_ideal": d["Rs"],
            "gamma_ideal": d["gamma"], "TIT": d["TIT"],
            # Station conditions (all passed through from design point)
            "T1": d["T1"], "P1": d["P1"],
            "T2": d["T2"], "P2": d["P2"],
            "T2p": d["T2p"],
            "T3": d["TIT"], "P3": d["Pc"],   # P3 = Pc (combustor inlet, corrected)
            "T4": d["T4"],  "P4": d["P4"],
            "T5": d["T5"],  "P5": d["P5"],
            # Component Cp
            "Cp_HPC": d["Cp_HPC"], "Cp_HPT": d["Cp_HPT"],
            # Component power [W]
            "P_HPC_W": P_HPC, "P_HPT_W": P_HPT,
            "P_H2T_W": P_H2T, "Power_compressor_W": P_comp,
            # Net power [W]
            "gaspath_net_W": gaspath_net, "h2_net_W": h2_net, "total_net_W": total_net,
            # HEX results
            "Q_tot_W": Q_tot, "T_hex_hot_in": T_hot_in, "T_hot_out": T_hot_out,
            "T_exh_final": d["T_exh_final"],
            "approach_min": approach_min, "approach_loc": approach_loc,
            "hex_feasible": approach_min > 0,
            # HEX sweep arrays (for plotting)
            "q": q, "T_cold": T_cold, "T_hot": T_hot, "approach": approach,
            # Recuperator
            "Q_regen_W": d["Q_regen_per_mf"] * mdot_f,
            # Efficiency
            "q_in_W": q_in, "eta_total": eta_total, "eta_gaspath": eta_gaspath,
            # Hydrogen state data (for T-S plotting)
            "h2_compressorout": d["h2_compressorout"],
            "P3_H2": d["P3_H2"], "h3_actual": d["h3_actual"], "TH3": d["TH3"],
        }

    # ------------------------------------------------------------------
    # Fuel-flow sizing (secant solver)
    # ------------------------------------------------------------------
    def _size_fuel_flow(self, tol=1.0, max_iter=50):
        """
        Find the fuel mass flow that meets P_target using the secant method.

        Because total net power is linear in mdot_f (the cycle is solved at a
        fixed design point), the secant method converges in effectively one step
        after the first two evaluations. The iteration is kept general in case
        non-linearities are introduced later (e.g. variable TIT).

        Returns
        -------
        mdot_f : float
            Converged fuel mass flow [kg/s].
        res : dict
            Full cycle results dict at the converged mdot_f.
        history : list of (mdot_f, residual) tuples
            Solver convergence history for reporting.
        """
        d = self._design

        def residual(mdot_f):
            res = self._run_cycle(mdot_f, d)
            return res["total_net_W"] - self.P_target, res

        history = []
        x0, x1 = self.mdot_f_init, self.mdot_f_init * 1.1
        f0, _   = residual(x0)
        f1, res = residual(x1)
        history.append((x0, f0))
        history.append((x1, f1))

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

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def report(self):
        """
        Print a formatted summary of the sized cycle to the terminal.

        Covers combustion parameters, fuel-flow solver convergence, recuperator
        performance, station thermodynamic conditions, H2 circuit power, and
        the overall efficiency / power budget.
        """
        if self.results is None:
            raise RuntimeError("Call size() before report().")

        res     = self.results
        history = self.history
        c       = self._console

        # --- Combustion / fuel balance ---
        c.print(Rule("[bold cyan]Combustion / Fuel Balance[/bold cyan]"))
        t = Table(box=box.ROUNDED, header_style="bold magenta", show_lines=True)
        t.add_column("Parameter",  style="cyan",   justify="left",  min_width=34)
        t.add_column("Value",      style="yellow",  justify="right", min_width=14)
        t.add_column("Units",      style="dim",     justify="left")
        t.add_row("Turbine Inlet Temperature",        f"{res['TIT']:.1f}",          "K")
        t.add_row("Combustor air inlet (T2')",        f"{res['T2p']:.1f}",          "K")
        t.add_row("O/F (energy balance at T2')",      f"{res['ideal_OF']:.2f}",     "-")
        t.add_row("Gamma (air, at T3)",               f"{res['gamma_ideal']:.4f}",  "-")
        if res["Rs_ideal"] is not None:
            t.add_row("Specific Gas Constant (CEA)", f"{res['Rs_ideal']:.2f}",     "J/(kg·K)")
        t.add_row("Total Mass-Flow Rate",             f"{res['mdot_tot']:.4f}",     "kg/s")
        c.print(t)

        # --- Fuel-flow solver convergence ---
        c.print(Rule("[bold cyan]Fuel-Flow Sizing[/bold cyan]"))
        t2 = Table(box=box.ROUNDED, header_style="bold green", show_lines=True)
        t2.add_column("Iter",           style="bold white", justify="center")
        t2.add_column("mdot_f [kg/s]",  style="yellow",     justify="right")
        t2.add_column("Residual [MW]",  style="cyan",        justify="right")
        for i, (m, f) in enumerate(history):
            t2.add_row(str(i), f"{m:.6f}", f"{f/1e6:+.4f}")
        c.print(t2)
        c.print(Panel(
            f"Target net power: [bold]{self.P_target/1e6:.3f} MW[/bold]\n"
            f"Sized fuel flow:  [bold green]{res['mdot_f']:.6f} kg/s[/bold green]",
            border_style="green",
        ))

        # --- Recuperator ---
        if self.USE_REGEN:
            c.print(Rule("[bold cyan]Recuperator[/bold cyan]"))
            t3 = Table(box=box.ROUNDED, header_style="bold yellow", show_lines=True)
            t3.add_column("Parameter", style="cyan",   justify="left",  min_width=32)
            t3.add_column("Value",     style="yellow",  justify="right", min_width=14)
            t3.add_column("Units",     style="dim",     justify="left")
            t3.add_row("Effectiveness",      f"{self.eta_regen:.2f}",              "-")
            t3.add_row("Air in (T2)",         f"{res['T2']:.1f}",                 "K")
            t3.add_row("Air out (T2')",       f"{res['T2p']:.1f}",                "K")
            t3.add_row("Recuperator duty",    f"{res['Q_regen_W']/1e6:.3f}",      "MW")
            t3.add_row("Final stack exhaust", f"{res['T_exh_final']:.1f}",        "K")
            c.print(t3)

        # --- Station table ---
        c.print(Rule("[bold cyan]Thermodynamic Cycle & Power[/bold cyan]"))
        t4 = Table(box=box.SIMPLE_HEAVY, header_style="bold blue", show_lines=True)
        t4.add_column("Station",     style="bold white", justify="center")
        t4.add_column("Description", style="cyan",       justify="left")
        t4.add_column("T [K]",       style="yellow",     justify="right")
        t4.add_column("P [bar]",     style="green",      justify="right")
        t4.add_row("1",  "Pre-HPC",             f"{res['T1']:.1f}",  f"{res['P1']:.3f}")
        t4.add_row("2",  "Post-HPC",            f"{res['T2']:.1f}",  f"{res['P2']:.3f}")
        t4.add_row("2'", "Post-recuperator",    f"{res['T2p']:.1f}", f"{res['P2']:.3f}")
        t4.add_row("3",  "Combustor inlet",     f"{res['T3']:.1f}",  f"{res['P3']:.3f}")
        t4.add_row("4",  "Post-CC / Pre-HPT",   f"{res['T4']:.1f}",  f"{res['P4']:.3f}")
        t4.add_row("5",  "Post-HPT",            f"{res['T5']:.1f}",  f"{res['P5']:.3f}")
        c.print(t4)

        # --- H2 HEX and expander ---
        c.print(Rule("[bold cyan]Hydrogen HEX & Expander Turbine[/bold cyan]"))
        t5 = Table(box=box.ROUNDED, header_style="bold yellow", show_lines=True)
        t5.add_column("Parameter", style="cyan",   justify="left",  min_width=32)
        t5.add_column("Value",     style="yellow",  justify="right", min_width=14)
        t5.add_column("Units",     style="dim",     justify="left")
        t5.add_row("Hot-side inlet to HEX",  f"{res['T_hex_hot_in']:.1f}",      "K")
        t5.add_row("Total heat transferred", f"{res['Q_tot_W']/1e6:.3f}",       "MW")
        t5.add_row("Exhaust temp post-HEX",  f"{res['T_hot_out']:.1f}",         "K")
        t5.add_row("Minimum approach temp",  f"{res['approach_min']:.1f}",      "K")
        c.print(t5)

        t6 = Table(box=box.ROUNDED, header_style="bold magenta", show_lines=True)
        t6.add_column("Component",   style="cyan",   justify="left",  min_width=30)
        t6.add_column("Power [kW]",  style="yellow",  justify="right", min_width=14)
        t6.add_row("GH2 Turbine (produced)",
                   f"[green]+{res['P_H2T_W']/1e3:.2f}[/green]")
        t6.add_row("H2 compressor (consumed)",
                   f"[red]-{res['Power_compressor_W']/1e3:.2f}[/red]")
        net_h2 = res["h2_net_W"] / 1e3
        nc = "green" if net_h2 >= 0 else "red"
        t6.add_row("[bold]Net GH2 circuit[/bold]",
                   f"[bold {nc}]{net_h2:+.2f}[/bold {nc}]")
        c.print(t6)

        # --- Overall summary ---
        c.print(Rule("[bold cyan]Overall Efficiency & Power Summary[/bold cyan]"))
        t7 = Table(box=box.DOUBLE_EDGE, header_style="bold white", show_lines=True)
        t7.add_column("Source", style="cyan",   justify="left",  min_width=28)
        t7.add_column("Value",  style="yellow",  justify="right", min_width=14)
        t7.add_row("Fuel power in (LHV)",   f"{res['q_in_W']/1e6:.3f} MW")
        t7.add_row("Gas-path efficiency",   f"{res['eta_gaspath']*100:.1f} %")
        t7.add_row("Total Efficiency",
                   f"[bold green]{res['eta_total']*100:.1f} %[/bold green]")
        t7.add_row("HPT output",
                   f"[green]+{res['P_HPT_W']/1e6:.3f} MW[/green]")
        t7.add_row("GH2 turbine output",
                   f"[green]+{res['P_H2T_W']/1e6:.3f} MW[/green]")
        t7.add_row("HPC demand",
                   f"[red]-{res['P_HPC_W']/1e6:.3f} MW[/red]")
        t7.add_row("H2 compressor demand",
                   f"[red]-{res['Power_compressor_W']/1e6:.3f} MW[/red]")
        total = res["total_net_W"] / 1e6
        tc = "green" if total >= 0 else "red"
        t7.add_row("[bold]TOTAL NET SHAFT[/bold]",
                   f"[bold {tc}]{total:+.3f} MW[/bold {tc}]")
        c.print(t7)

    # ------------------------------------------------------------------
    # T-S plotting helpers (private)
    # ------------------------------------------------------------------
    @staticmethod
    def _isobar(P_bar, T_start, T_end, fluid_name, num_points=60):
        """
        Compute a real-fluid isobaric curve in T-s space between two temperatures.

        Used to draw thermodynamically accurate process lines on T-S diagrams
        instead of straight tie-lines (which are only correct for ideal gases
        with constant cp).
        """
        if abs(T_start - T_end) < 0.1:
            return np.array([]), np.array([])
        T_arr = np.linspace(T_start, T_end, num_points)
        S_arr = np.array([
            PropsSI('S', 'P', P_bar*1e5, 'T', t, fluid_name) for t in T_arr
        ])
        return S_arr, T_arr

    def _h2_state_points(self, res):
        """
        Compute the four hydrogen circuit state points for T-S plotting.

        Returns a list of (label, P_bar, T, s) tuples for stations HA--HD.
        The path is open: HD feeds into the combustor and does not return to HA.
        """
        PA, TA = self.P_pre_comp, self.T_pre_comp
        sA = PropsSI('S', 'P', PA*1e5, 'T', TA, self.fluid)

        PB  = self.PH1
        hB  = res['h2_compressorout']
        TB  = PropsSI('T', 'P', PB*1e5, 'H', hB, self.fluid)
        sB  = PropsSI('S', 'P', PB*1e5, 'H', hB, self.fluid)

        PC, TC = self.PH1, self.TH2
        sC  = PropsSI('S', 'P', PC*1e5, 'T', TC, self.fluid)

        PD  = res['P3_H2']
        hD  = res['h3_actual']
        TD  = res['TH3']
        sD  = PropsSI('S', 'P', PD*1e5, 'H', hD, self.fluid)

        return [("A", PA, TA, sA), ("B", PB, TB, sB),
                ("C", PC, TC, sC), ("D", PD, TD, sD)]

    def _draw_h2_path(self, ax, states, label_prefix=""):
        """
        Draw the open hydrogen path HA->HB->HC->HD on a given matplotlib axis.

        Compression (HA->HB) and turbine expansion (HC->HD) are drawn as
        straight tie-lines since the exact entropy path is process-dependent.
        HEX heating (HB->HC) is drawn as a real isobar.
        """
        (nA, PA, TA, sA), (nB, PB, TB, sB), (nC, PC, TC, sC), (nD, PD, TD, sD) = states

        # HA -> HB: compression (tie-line)
        ax.plot([sA, sB], [TA, TB], '-', color='steelblue', linewidth=2,
                label=f"{label_prefix}H2 compression")

        # HB -> HC: isobaric HEX heating (real fluid curve)
        S_hex, T_hex = self._isobar(PB, TB, TC, self.fluid)
        if S_hex.size:
            ax.plot(S_hex, T_hex, '-', color='crimson', linewidth=2.5,
                    label=f"{label_prefix}H2 HEX heating")

        # HC -> HD: expander turbine (tie-line)
        ax.plot([sC, sD], [TC, TD], '-', color='seagreen', linewidth=2,
                label=f"{label_prefix}H2 turbine")

        for name, P, T, s in states:
            ax.plot(s, T, 'ko', markersize=5, zorder=5)
            ax.annotate(f" H{name}", (s, T), fontsize=10,
                        xytext=(5, 2), textcoords="offset points")

        # Label the open-path exit
        ax.annotate("to combustor", (sD, TD), fontsize=8, color='seagreen',
                    xytext=(8, -12), textcoords="offset points")

    # ------------------------------------------------------------------
    # Plotting methods (public)
    # ------------------------------------------------------------------
    def plot_ts(self):
        """
        Plot the air-path T-S diagram for the main gas cycle.

        Process lines are drawn as real-fluid isobars where applicable.
        The recuperator heat transfer arrow shows the coupling between the
        exhaust cooling and the air preheating streams.
        """
        if self.results is None:
            raise RuntimeError("Call size() before plot_ts().")

        res = self.results
        plt.figure(figsize=(9, 7))

        P1, T1           = res['P1'], res['T1']
        P2, T2, T2p      = res['P2'], res['T2'], res['T2p']
        P4, T4           = res['P4'], res['T4']
        P5, T5           = res['P5'], res['T5']
        T_exh            = res['T_exh_final']

        s1   = PropsSI('S', 'P', P1*1e5, 'T', T1,   'Air')
        s2   = PropsSI('S', 'P', P2*1e5, 'T', T2,   'Air')
        s2p  = PropsSI('S', 'P', P2*1e5, 'T', T2p,  'Air')
        s4   = PropsSI('S', 'P', P4*1e5, 'T', T4,   'Air')
        s5   = PropsSI('S', 'P', P5*1e5, 'T', T5,   'Air')
        s_exh = PropsSI('S', 'P', P5*1e5, 'T', T_exh, 'Air')

        # 1 -> 2: HPC compression (tie-line)
        plt.plot([s1, s2], [T1, T2], 'k-', linewidth=1.5,
                 label='Compression / Expansion')

        # 2 -> 2': Recuperator air heating (real isobar)
        Sr, Tr = self._isobar(P2, T2, T2p, 'Air')
        if Sr.size:
            plt.plot(Sr, Tr, color='orange', linewidth=3,
                     label="Recuperator (Air Heating)")

        # 2' -> 4: Combustor heat addition (real isobar)
        Sc, Tc = self._isobar(P4, T2p, T4, 'Air')
        plt.plot(Sc, Tc, color='red', linewidth=2, label="Combustor (Heat Addition)")

        # 4 -> 5: HPT expansion (tie-line)
        plt.plot([s4, s5], [T4, T5], 'k-', linewidth=1.5)

        # 5 -> T_exh: Exhaust cooling through recuperator + H2 HEX (real isobar)
        Se, Te = self._isobar(P5, T5, T_exh, 'Air')
        plt.plot(Se, Te, color='purple', linewidth=3,
                 label="Exhaust (Regen + HEX Sink)")

        # T_exh -> 1: Atmospheric rejection (tie-line closes the loop)
        # Direct tie-line handles the pressure drop from P5 to P1 without
        # attempting a physically ambiguous isobaric closure.
        plt.plot([s_exh, s1], [T_exh, T1], color='gray', linestyle='--',
                 linewidth=1.5, label="Atmospheric Rejection")

        # Recuperator heat-transfer arrow (visual coupling, not to scale)
        if T2p > T2:
            T_cold_mid = (T2 + T2p) / 2
            s_cold_mid = PropsSI('S', 'P', P2*1e5, 'T', T_cold_mid, 'Air')
            T_hot_mid  = T5 - (T2p - T2) / 2
            s_hot_mid  = PropsSI('S', 'P', P5*1e5, 'T', T_hot_mid, 'Air')
            plt.annotate('', xy=(s_cold_mid, T_cold_mid),
                         xytext=(s_hot_mid, T_hot_mid),
                         arrowprops=dict(arrowstyle='->', color='orange', lw=2))
            plt.text((s_cold_mid + s_hot_mid)/2,
                     (T_cold_mid + T_hot_mid)/2 + 25,
                     'Regen Heat Transfer',
                     color='orange', ha='center', fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        # Station markers
        stations = {
            "1": (s1, T1), "2": (s2, T2), "2'": (s2p, T2p),
            "4": (s4, T4), "5": (s5, T5), "Exh": (s_exh, T_exh),
        }
        for name, (s, t) in stations.items():
            if name == "2'" and T2p == T2:
                continue
            plt.plot(s, t, 'ko', markersize=5, zorder=5)
            plt.annotate(f" {name}", (s, t), fontsize=10,
                         xytext=(5, 2), textcoords="offset points")

        plt.title("Gas-Path T-S Diagram")
        plt.xlabel("Specific Entropy, s [J/kg·K]")
        plt.ylabel("Temperature, T [K]")
        plt.grid(True, alpha=0.5)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    def plot_ts_h2(self):
        """
        Plot the hydrogen fuel circuit T-S diagram (ParaHydrogen working fluid).

        The entropy scale here belongs to hydrogen and is NOT comparable to the
        air-side diagram. The path is open: HA->HB->HC->HD, then fuel enters
        the combustor.
        """
        if self.results is None:
            raise RuntimeError("Call size() before plot_ts_h2().")

        states = self._h2_state_points(self.results)
        plt.figure(figsize=(9, 7))
        self._draw_h2_path(plt.gca(), states)
        plt.title("Hydrogen-Circuit T-S Diagram (ParaHydrogen)")
        plt.xlabel("Specific Entropy, s [J/kg·K]  (hydrogen reference)")
        plt.ylabel("Temperature, T [K]")
        plt.grid(True, alpha=0.5)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    def plot_ts_overlay(self):
        """
        Overlay the air gas path and hydrogen circuit on a shared temperature axis.

        The two working fluids have incompatible entropy reference states and
        live on very different numerical scales (air ~4-5 kJ/kg/K, H2 ~24-56
        kJ/kg/K). Plotting both on a single s-axis would be physically meaningless,
        so the hydrogen circuit uses an independent top x-axis via twiny().
        Only the temperature (y) axis is shared and physically comparable.
        """
        if self.results is None:
            raise RuntimeError("Call size() before plot_ts_overlay().")

        res = self.results
        fig, ax_air = plt.subplots(figsize=(11, 7))

        # --- Air gas path (bottom x-axis) ---
        P1, T1           = res['P1'], res['T1']
        P2, T2, T2p      = res['P2'], res['T2'], res['T2p']
        P4, T4           = res['P4'], res['T4']
        P5, T5           = res['P5'], res['T5']
        T_exh            = res['T_exh_final']

        s1    = PropsSI('S', 'P', P1*1e5, 'T', T1,    'Air')
        s2    = PropsSI('S', 'P', P2*1e5, 'T', T2,    'Air')
        s2p   = PropsSI('S', 'P', P2*1e5, 'T', T2p,   'Air')
        s4    = PropsSI('S', 'P', P4*1e5, 'T', T4,    'Air')
        s5    = PropsSI('S', 'P', P5*1e5, 'T', T5,    'Air')
        s_exh = PropsSI('S', 'P', P5*1e5, 'T', T_exh, 'Air')

        ax_air.plot([s1, s2], [T1, T2], 'k-', lw=1.5,
                    label="Air compression / expansion")
        Sr, Tr = self._isobar(P2, T2, T2p, 'Air')
        if Sr.size:
            ax_air.plot(Sr, Tr, color='orange', lw=3,
                        label="Air recuperator heating")
        Sc, Tc = self._isobar(P4, T2p, T4, 'Air')
        ax_air.plot(Sc, Tc, color='red', lw=2, label="Air combustor")
        ax_air.plot([s4, s5], [T4, T5], 'k-', lw=1.5)
        Se, Te = self._isobar(P5, T5, T_exh, 'Air')
        ax_air.plot(Se, Te, color='purple', lw=3,
                    label="Air exhaust (regen + HEX)")
        ax_air.plot([s_exh, s1], [T_exh, T1], color='gray', ls='--', lw=1.5,
                    label="Atmospheric rejection")

        for name, s, t in [("1",  s1,   T1),   ("2",  s2,  T2),
                            ("2'", s2p,  T2p),  ("4",  s4,  T4),
                            ("5",  s5,   T5),   ("Exh", s_exh, T_exh)]:
            if name == "2'" and T2p == T2:
                continue
            ax_air.plot(s, t, 'ko', ms=5, zorder=5)
            ax_air.annotate(f" {name}", (s, t), fontsize=9,
                            xytext=(4, 2), textcoords="offset points")

        ax_air.set_xlabel("Air specific entropy, s [J/kg·K]  (air reference)")
        ax_air.set_ylabel("Temperature, T [K]")
        ax_air.grid(True, alpha=0.4)

        # --- Hydrogen circuit (top x-axis, independent entropy scale) ---
        ax_h2 = ax_air.twiny()
        states = self._h2_state_points(res)
        self._draw_h2_path(ax_h2, states)
        ax_h2.set_xlabel(
            "Hydrogen specific entropy, s [J/kg·K]  "
            "(hydrogen reference -- NOT comparable to air scale)",
            color='dimgray',
        )

        ax_air.set_title(
            "Overlay: Air Gas Path + Hydrogen Circuit\n"
            "(shared temperature axis; entropy axes are independent)"
        )

        # Merge legends from both axes
        h1, l1 = ax_air.get_legend_handles_labels()
        h2, l2 = ax_h2.get_legend_handles_labels()
        ax_air.legend(h1 + h2, l1 + l2,
                      loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0.)
        fig.tight_layout()
        plt.show()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    cea = CEA_Obj(
        oxName="AIR", fuelName="GH2",
        pressure_units="bar", temperature_units="K", isp_units="sec",
    )

    engine = GasTurbineCycle()      # all defaults match original sizing target
    engine.size(cea=cea)
    engine.report()
    engine.plot_ts()
    engine.plot_ts_h2()
    engine.plot_ts_overlay()