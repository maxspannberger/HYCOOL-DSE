"""
ClassII_Scissor_Plot.py

Scissor plot calculation adapted from the original standalone Scissor_Plot.py.

The equations are intentionally kept close to the original file. The main
change is that aircraft geometry is pulled from the converged Class II result:
    - wing area, span, root chord, tip chord and MAC from result.iteration_log[-1]
    - horizontal tail area and span from result.tail_rechecked
    - mass and landing CL from the final Class II result

Values not currently available in AircraftConfig or ClassIIResult are kept as
explicit assumptions in ScissorPlotInput, so they are easy to find and replace.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Allow running both from inside WeightEstimations and from the project root.
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

try:
    from WeightEstimations.Aircraft_Config import AircraftConfig, default_q400_hycool
    from WeightEstimations.mainClassII import ClassIIResult, run_class_ii
    from WeightEstimations.ISA import isa
    from General.component_parameters import component_params as comp_params
except ImportError:  # fallback for running the file directly inside WeightEstimations
    from Aircraft_Config import AircraftConfig, default_q400_hycool
    from mainClassII import ClassIIResult, run_class_ii
    from ISA import isa
    from General.component_parameters import component_params as comp_params


G = 9.80665


def sweep_at_chord_fraction(
    sweep_known: float,
    x_known: float,
    x_target: float,
    aspect_ratio: float,
    taper: float,
) -> float:
    """
    Convert sweep from one chord fraction to another for a trapezoidal wing.

    x = 0.25 gives quarter-chord sweep.
    x = 0.50 gives half-chord sweep.

    tan(Lambda_x2) = tan(Lambda_x1)
                      + 4/AR * (x1 - x2) * (1 - taper)/(1 + taper)
    """
    return np.arctan(
        np.tan(sweep_known)
        + (4.0 / aspect_ratio) * (x_known - x_target) * (1.0 - taper) / (1.0 + taper)
    )


@dataclass
class ScissorPlotInput:
    # ---------------- Aircraft geometry from converged Class II result ----------------
    bf: float                       # fuselage width or equivalent diameter [m]
    hf: float                       # fuselage height or equivalent diameter [m]
    lf: float                       # fuselage length after tank iteration [m]
    lfn: float                      # distance from fuselage nose to wing root/reference [m]

    S: float                        # wing reference area after MTOW iteration [m^2]
    Sh: float                       # horizontal stabilizer area after tail recheck [m^2]
    b: float                        # wing span after MTOW iteration [m]
    c: float                        # wing MAC after MTOW iteration [m]
    ct: float                       # wing tip chord after MTOW iteration [m]
    cr: float                       # wing root chord after MTOW iteration [m]

    sweep25: float                  # wing quarter-chord sweep [rad]
    sweep50: float                  # wing half-chord sweep [rad]
    sweep25_tail: float             # horizontal tail quarter-chord sweep [rad]
    sweep50_tail: float             # horizontal tail half-chord sweep [rad]

    bn: float                       # nacelle or propulsor equivalent diameter [m]
    ln: float                       # nacelle longitudinal arm relative to wing a.c. [m]
    bh: float                       # horizontal tail span after tail recheck [m]
    lh: float                       # horizontal tail arm [m]
    hh: float                       # vertical distance between wing plane and horizontal tail plane [m]

    Mcruise: float                  # cruise Mach number for stability condition [-]
    Vlanding: float                 # landing or approach speed [m/s]
    M_max_landing: float            # landing mass used for CL calculation [kg]

    # ---------------- Controllability inputs from old scissor plot ----------------
    CL0: float = 1.4                # CL at zero AoA for flapped wing
    mu1: float = 0.215
    mu2: float = 0.79
    mu3: float = 0.045
    cdash_c: float = 1.2917
    delta_cl_max: float = 1.8
    Swf_S: float = 0.621113839
    Cm0_airfoil: float = -0.11

    # ---------------- CG range ----------------
    # Replace these with the final loading diagram output when available.
    xcg_lower: float = 0.283 * 0.98
    xcg_upper: float = 0.640 * 1.02

    # ---------------- Constants from lecture/literature ----------------
    kn: float = -4.0                # nacelle empirical factor, slide 40
    Vh_V: float = 0.85              # tail/wing speed ratio, slide 42
    etah: float = 0.95              # horizontal tail efficiency, slide 43 comments
    CLh: float = -0.8               # adjustable horizontal tail
    gamma: float = 1.4
    R: float = 287.0
    T: float = 288.15
    S_M: float = 0.05               # stability margin
    xac_w: float = 0.25             # assumed wing aerodynamic centre wrt LEMAC

    # ---------------- Optional modelling assumptions ----------------
    winglet_AR_increment: float = 0.0

    @classmethod
    def from_class_ii(
        cls,
        cfg: AircraftConfig,
        result: ClassIIResult,
        *,
        xcg_lower: Optional[float] = None,
        xcg_upper: Optional[float] = None,
        lfn: Optional[float] = None,
        bn: Optional[float] = None,
        ln: Optional[float] = None,
        tail_taper: Optional[float] = None,
        Vlanding: Optional[float] = None,
        M_max_landing: Optional[float] = None,
        CL0: float = 1.4,
        mu1: float = 0.215,
        mu2: float = 0.79,
        mu3: float = 0.045,
        cdash_c: float = 1.2917,
        delta_cl_max: float = 1.8,
        Swf_S: float = 0.621113839,
        Cm0_airfoil: float = -0.11,
    ) -> "ScissorPlotInput":
        """
        Build scissor plot inputs from the converged Class II output.

        This method deliberately uses values after the MTOW loop:
            final = result.iteration_log[-1]
            result.Wing_Area, result.Wing_span
            result.tail_rechecked.S_h, result.tail_rechecked.b_h

        Inputs that are not available in the current project structure can be
        overridden here, for example xcg_lower, xcg_upper, bn, ln and lfn.
        """
        if not result.iteration_log:
            raise ValueError("result.iteration_log is empty. Run run_class_ii before building the scissor plot input.")

        final = result.iteration_log[-1]

        S = float(result.Wing_Area)
        b = float(result.Wing_span)
        c = float(final["MAC_m"])
        cr = float(final["c_root_m"])
        ct = float(final["c_tip_m"])

        Sh = float(result.tail_rechecked.S_h)
        bh = float(result.tail_rechecked.b_h)

        # Use the final equivalent fuselage diameter from the iteration log.
        # The current ClassIIResult stores l_f_m and d_f_m, but not separate b_f and h_f.
        # For the current circular fuselage this is equivalent. If b_f != h_f later,
        # pass bf and hf explicitly by extending this input class.
        d_f_final = float(final.get("d_f_m", cfg.d_f))
        bf = d_f_final
        hf = d_f_final
        lf = float(result.l_f_m)

        # Current config does not store the wing longitudinal location directly.
        # lfn is estimated from fuselage length minus tail arm unless provided.
        lfn_use = float(lfn) if lfn is not None else max(lf - cfg.l_t, 0.0)

        # Current config does not store nacelle diameter and nacelle longitudinal arm separately.
        # bn defaults to propfan disk diameter. ln defaults to 0, meaning no nacelle a.c. shift.
        # Replace these if you have better nacelle geometry.
        bn_use = float(bn) if bn is not None else float(cfg.D_propfan)
        ln_use = float(ln) if ln is not None else 0.0

        wing_taper = cfg.taper
        tail_taper_use = float(tail_taper) if tail_taper is not None else cfg.taper

        sweep50 = float(cfg.sweep_half)
        sweep25 = sweep_at_chord_fraction(
            sweep_known=sweep50,
            x_known=0.50,
            x_target=0.25,
            aspect_ratio=cfg.AR,
            taper=wing_taper,
        )

        sweep50_tail = float(cfg.sweep_h_half)
        AR_h = bh**2 / Sh
        sweep25_tail = sweep_at_chord_fraction(
            sweep_known=sweep50_tail,
            x_known=0.50,
            x_target=0.25,
            aspect_ratio=AR_h,
            taper=tail_taper_use,
        )

        Vlanding_use = float(Vlanding) if Vlanding is not None else 1.3 * cfg.V_stall
        M_max_landing_use = float(M_max_landing) if M_max_landing is not None else result.MZFW

        kwargs = dict(
            bf=bf,
            hf=hf,
            lf=lf,
            lfn=lfn_use,
            S=S,
            Sh=Sh,
            b=b,
            c=c,
            ct=ct,
            cr=cr,
            sweep25=sweep25,
            sweep50=sweep50,
            sweep25_tail=sweep25_tail,
            sweep50_tail=sweep50_tail,
            bn=bn_use,
            ln=ln_use,
            bh=bh,
            lh=float(cfg.l_h),
            hh=float(cfg.h_h),
            Mcruise=float(cfg.M_cruise),
            Vlanding=Vlanding_use,
            M_max_landing=M_max_landing_use,
            CL0=CL0,
            mu1=mu1,
            mu2=mu2,
            mu3=mu3,
            cdash_c=cdash_c,
            delta_cl_max=delta_cl_max,
            Swf_S=Swf_S,
            Cm0_airfoil=Cm0_airfoil,
        )

        if xcg_lower is not None:
            kwargs["xcg_lower"] = float(xcg_lower)
        if xcg_upper is not None:
            kwargs["xcg_upper"] = float(xcg_upper)

        return cls(**kwargs)


@dataclass
class ScissorPlotBreakdown:
    xcg: np.ndarray
    ShS_s: np.ndarray
    ShS_s_no_margin: np.ndarray
    ShS_c: np.ndarray
    actual_ShS: float

    # Stability intermediates
    CLalpha_h_s: float
    CLalpha_w_s: float
    CLalpha_A_h_s: float
    xacf1_s: float
    xacf2_s: float
    xacn_s: float
    xac_s: float
    depsilon_dalpha: float

    # Controllability intermediates
    CLalpha_h_c: float
    CLalpha_w_c: float
    CLalpha_A_h_c: float
    xacf1_c: float
    xacf2_c: float
    xacn_c: float
    xac_c: float
    Cmac_w: float
    Cmac_fus: float
    delta_Cmac: float
    Cmac_f: float
    Cmac: float
    CLA_h: float
    CL: float
    A: float
    Ah: float
    S_net: float


class ScissorPlotEstimator:
    def __init__(self, inputs: ScissorPlotInput):
        self.i = inputs
        self._validate()

    def _validate(self) -> None:
        d = self.i
        required = dict(
            bf=d.bf, hf=d.hf, lf=d.lf, lfn=d.lfn,
            S=d.S, Sh=d.Sh, b=d.b, c=d.c, ct=d.ct, cr=d.cr,
            bh=d.bh, lh=d.lh, hh=d.hh,
            Mcruise=d.Mcruise, Vlanding=d.Vlanding, M_max_landing=d.M_max_landing,
        )
        missing = [name for name, value in required.items() if value <= 0]
        if missing:
            raise ValueError(f"Scissor plot inputs not set or zero: {missing}")

    def compute(self) -> ScissorPlotBreakdown:
        d = self.i

        # Keep the same variable names and equation order as the original script.
        kn = d.kn
        Vh_V = d.Vh_V
        etah = d.etah
        CLh = d.CLh
        gamma = d.gamma
        R = d.R
        T = d.T
        S_M = d.S_M

        xac_w = d.xac_w
        bf = d.bf
        hf = d.hf
        lf = d.lf
        lfn = d.lfn
        S = d.S
        Sh = d.Sh
        b = d.b
        c = d.c
        ct = d.ct
        cr = d.cr
        sweep25 = d.sweep25
        sweep50 = d.sweep50
        sweep25_tail = d.sweep25_tail
        sweep50_tail = d.sweep50_tail
        bn = d.bn
        ln = d.ln
        bh = d.bh
        Ah = bh**2 / Sh
        A = b**2 / S + d.winglet_AR_increment
        Mcruise = d.Mcruise
        S_net = S - bf * cr
        lh = d.lh
        hh = d.hh
        CL0 = d.CL0
        mu1 = d.mu1
        mu2 = d.mu2
        mu3 = d.mu3
        cdash_c = d.cdash_c
        delta_cl_max = d.delta_cl_max
        Swf_S = d.Swf_S
        Cm0_airfoil = d.Cm0_airfoil
        Vlanding = d.Vlanding
        M_max_landing = d.M_max_landing

        # Landing lift coefficient from final landing mass and final wing area.
        # This replaces the old hard-coded CL = 1.661 while preserving the equation usage.
        rho_landing = 1.225
        CL = M_max_landing * G / (0.5 * rho_landing * Vlanding**2 * S)
        CLA_h = CL

        # ------------------------------------- STABILITY CONDITION --------------------------------------------------

        # CLalpha_h -> CL gradient for horizontal tail
        betas = np.sqrt(1 - Mcruise**2)
        CLalpha_h_s = (2*np.pi*Ah)/(2 + np.sqrt(4 + (Ah*betas/etah)**2 * (1+ np.tan(sweep50_tail)**2/betas**2)))

        # CLalpha_A_h -> CL gradient for fuselage + wing minus tail
        CLalpha_w_s = (2*np.pi*A)/(2 + np.sqrt(4 + (A*betas/etah)**2 * (1+ np.tan(sweep50)**2/betas**2)))
        CLalpha_A_h_s = CLalpha_w_s * (1 + 2.15 * bf/b)*S_net/S + np.pi/2 * bf**2 / S

        # xac -> distance aerodynamic center wrt LEMAC
        xacf1_s = (-1.8/CLalpha_A_h_s) * (bf*hf*lfn) / (S*c)
        xacf2_s = (0.273/(1 + ct/cr))*((bf*S/b * (b-bf))/(c**2*(b + 2.15*bf)))*np.tan(sweep25)
        xacn_s = 2 * (kn * (bn**2 * ln)/(S*c*CLalpha_A_h_s))
        xac_s = xac_w + xacf1_s + xacf2_s + xacn_s

        # depsilon_dalpha -> downwash gradient
        r = 2*lh/b
        mtv = 2 * hh / b
        Kepsilon_sweep = (0.1124 + 0.1265 * sweep25 + 0.1766 * sweep25**2)/(r**2) + 0.1024/r + 2
        Kepsilon_sweep0 = 0.1124/r**2 + 0.1024/r + 2
        depsilon_dalpha = Kepsilon_sweep/Kepsilon_sweep0 * ((r/(r**2+mtv**2))*0.4876/np.sqrt(r**2 + 0.6319+mtv**2) + (1 + (r**2/(r**2+0.7915+5.0734*(mtv**2)))**0.3113)*(1 - np.sqrt(mtv**2/(1+mtv**2))))*CLalpha_w_s/(np.pi*A)

        # we need to compute Shs for each xcg
        xcg = np.arange(-0.4, 1.2, 0.01)

        # Stability Sh/S
        denom_s = CLalpha_h_s/CLalpha_A_h_s * (1 - depsilon_dalpha)*lh/c * (Vh_V)**2
        ShS_s = xcg/denom_s - (xac_s - S_M)/denom_s
        ShS_s_no_margin = xcg/denom_s - (xac_s)/denom_s

        # ------------------------------------- CONTROLLABILITY CONDITION --------------------------------------------------

        # CLalpha_h -> CL gradient for horizontal tail
        Mlanding = Vlanding / np.sqrt(gamma*R*T)
        betac = np.sqrt(1 - Mlanding**2)
        CLalpha_h_c = (2*np.pi*Ah)/(2 + np.sqrt(4 + (Ah*betac/etah)**2 * (1+ np.tan(sweep50_tail)**2/betac**2)))

        # CLalpha_A_h -> CL gradient for fuselage + wing minus tail
        CLalpha_w_c = (2*np.pi*A)/(2 + np.sqrt(4 + (A*betac/etah)**2 * (1+ np.tan(sweep50)**2/betac**2)))
        CLalpha_A_h_c = CLalpha_w_c * (1 + 2.15 * bf/b)*S_net/S + np.pi/2 * bf**2 / S

        # xac -> distance aerodynamic center wrt LEMAC
        xacf1_c = (-1.8/CLalpha_A_h_c) * (bf*hf*lfn) / (S*c)
        xacf2_c = (0.273/(1 + ct/cr))*((bf*S/b * (b-bf))/(c**2*(b + 2.15*bf)))*np.tan(sweep25)
        xacn_c = 2 * (kn * (bn**2 * ln)/(S*c*CLalpha_A_h_c))
        xac_c = xac_w + xacf1_c + xacf2_c + xacn_c

        # Cmac calculation
        Cmac_w = Cm0_airfoil * (A * np.cos(sweep25)**2 / (A + 2*np.cos(sweep25)))
        Cmac_fus = -1.8*(1-2.5*bf/lf)*np.pi*bf*hf*lf/(4*S*c) * CL0/CLalpha_A_h_c
        delta_Cmac = mu2 * (-mu1 * delta_cl_max * cdash_c - (CL + delta_cl_max*(1-Swf_S))*1/8 * cdash_c*(cdash_c-1)) + 0.7*A*mu3/(1+2/A) * delta_cl_max*np.tan(sweep25)
        Cmac_f = delta_Cmac - CL * (0.25 - xac_c)
        Cmac = Cmac_w + Cmac_f + Cmac_fus

        # Controllability Sh/S
        denom_c = CLh/CLA_h * lh/c * (Vh_V)**2
        ShS_c = xcg / denom_c + (Cmac/CLA_h - xac_c)/denom_c

        # Actual aircraft Sh/S based on final tail recheck and final wing area.
        actual_ShS = Sh / S

        return ScissorPlotBreakdown(
            xcg=xcg,
            ShS_s=ShS_s,
            ShS_s_no_margin=ShS_s_no_margin,
            ShS_c=ShS_c,
            actual_ShS=actual_ShS,
            CLalpha_h_s=CLalpha_h_s,
            CLalpha_w_s=CLalpha_w_s,
            CLalpha_A_h_s=CLalpha_A_h_s,
            xacf1_s=xacf1_s,
            xacf2_s=xacf2_s,
            xacn_s=xacn_s,
            xac_s=xac_s,
            depsilon_dalpha=depsilon_dalpha,
            CLalpha_h_c=CLalpha_h_c,
            CLalpha_w_c=CLalpha_w_c,
            CLalpha_A_h_c=CLalpha_A_h_c,
            xacf1_c=xacf1_c,
            xacf2_c=xacf2_c,
            xacn_c=xacn_c,
            xac_c=xac_c,
            Cmac_w=Cmac_w,
            Cmac_fus=Cmac_fus,
            delta_Cmac=delta_Cmac,
            Cmac_f=Cmac_f,
            Cmac=Cmac,
            CLA_h=CLA_h,
            CL=CL,
            A=A,
            Ah=Ah,
            S_net=S_net,
        )

    def plot(
        self,
        breakdown: Optional[ScissorPlotBreakdown] = None,
        *,
        show: bool = True,
        save_path: Optional[str | Path] = None,
    ) -> ScissorPlotBreakdown:
        d = self.i
        bd = breakdown if breakdown is not None else self.compute()

        plt.figure(figsize=(10, 6))

        # Plot the primary equation lines
        plt.plot(bd.xcg, bd.ShS_s, 'b-', label='Stability Line With Safety Margin')
        plt.plot(bd.xcg, bd.ShS_c, 'g-', label='Controllability Line')
        plt.plot(bd.xcg, bd.ShS_s_no_margin, 'k:', label='Neutral Stability Curve')

        # Shade the invalid regions red
        invalid_ShS = np.maximum(bd.ShS_s, bd.ShS_c)
        plt.fill_between(bd.xcg, 0, invalid_ShS, color='red', alpha=0.4)

        # Plot actual center of gravity range
        plt.plot(
            [d.xcg_lower, d.xcg_upper],
            [bd.actual_ShS, bd.actual_ShS],
            'k-',
            linewidth=2.5,
            label=f'Actual CG Range (Sh/S = {bd.actual_ShS:.3f})',
        )

        # End caps
        plt.plot([d.xcg_lower, d.xcg_lower], [bd.actual_ShS - 0.005, bd.actual_ShS + 0.005], 'k-', linewidth=2.5)
        plt.plot([d.xcg_upper, d.xcg_upper], [bd.actual_ShS - 0.005, bd.actual_ShS + 0.005], 'k-', linewidth=2.5)

        # Plot formatting
        plt.title('Scissor Plot of the Aircraft')
        plt.xlabel('Xcg / MAC')
        plt.ylabel('Sh / S')
        plt.xlim(0.0, 1.1)
        plt.ylim(0.0, max(0.3, 1.1 * bd.actual_ShS))
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        return bd

    def print_debug(self, bd: Optional[ScissorPlotBreakdown] = None) -> None:
        d = self.i
        bd = bd if bd is not None else self.compute()
        print(bd.depsilon_dalpha)
        print(f'xac control {bd.xac_c}, xac stab {bd.xac_s}')
        print(f'cruise xac wing {d.xac_w}, cruise xac fus {bd.xacf1_s+bd.xacf2_s}, cruise xac nac {bd.xacn_s}')
        print(f'appr xac wing {d.xac_w}, appr xac fus {bd.xacf1_c+bd.xacf2_c}, appr xac nac {bd.xacn_c}')
        print(f'Cmac {bd.Cmac}')
        print(f'Cmac wing {bd.Cmac_w}, Cmac fus {bd.Cmac_fus}, Cmac flaps {bd.Cmac_f}')
        print(f'CLh {d.CLh}')
        print(f'CLA_h {bd.CLA_h}')
        print(f'lh {d.lh}')
        print(f'Vh/V {d.Vh_V}')
        print(f'CLalpha_h control {bd.CLalpha_h_c}, CLalpha_h stab {bd.CLalpha_h_s}')
        print(f'CLalpha_a_h cruise contribution wing {bd.CLalpha_w_s}, contribution fuselage {bd.CLalpha_A_h_s-bd.CLalpha_w_s}')
        print(f'CLalpha_A_h control {bd.CLalpha_A_h_c}, CLalpha_A_h stab {bd.CLalpha_A_h_s}')
        print(f'downwash gradient {bd.depsilon_dalpha}')
        print(f'wing aspect ratio {bd.A}')
        print(f'actual Sh/S {bd.actual_ShS}')


if __name__ == "__main__":
    cfg = default_q400_hycool()
    result = run_class_ii(cfg, comp=comp_params, tol=1.0, max_iter=100, verbose=True)

    inp = ScissorPlotInput.from_class_ii(
        cfg,
        result,
        # Replace these with loading diagram outputs when available.
        xcg_lower=0.283 * 0.98,
        xcg_upper=0.640 * 1.02,
        # Replace these with actual nacelle geometry when available.
        ln=0.0,
    )

    estimator = ScissorPlotEstimator(inp)
    breakdown = estimator.compute()
    estimator.print_debug(breakdown)
    estimator.plot(breakdown, show=True, save_path="scissor_plot.png")
