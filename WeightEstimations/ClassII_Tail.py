"""
Tail_Sizing.py

Class II control surface sizing.

Sizing logic:

    S_h = max( volume-coefficient stability sizing , elevator-trim sizing )
    S_v = max( volume-coefficient stability sizing , OEI rudder-control sizing )

Why: for prop / propfan aircraft the cruise weathercock-stability requirement
(captured by V_v) is often LESS demanding than the low-speed OEI control
requirement (CS-25.149). If you only size on V_v you get a tail that is
statically stable in cruise but cannot hold heading after engine failure on
takeoff. The Q400 vertical tail (b_v ~ 8 m) is so much bigger than V_v alone
predicts because OEI control is what actually drives it.

For the horizontal tail, volume-coefficient sizing usually wins, but a quick
elevator-power check is included for completeness so high-speed dive trim
or low-speed flap-down trim doesn't quietly violate.

S_v sizing per Torenbeek's OEI rudder formulation:
    M_engine     = T_TO * y_engine
    M_rudder_max = k_r * S_r * l_v * q_mc * delta_r_max
    Equilibrium: M_rudder_max >= M_engine
    With S_r = (Sr_Sv_max) * S_v this gives the minimum S_v for control.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path

# Allow running both from inside WeightEstimations and from the project root.
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))
from WeightEstimations.ISA import isa
from WeightEstimations.Aircraft_Config import AircraftConfig

from rich.table import Table
from rich.console import Group
from rich.text import Text
from rich import print

@dataclass
class TailSizing_Input:

    # Wing reference geometry
    S_ref:          float = 0.0
    MAC:            float = 0.0
    b:              float = 0.0
    AR:             float = 0.0

    # Tail moment arms
    l_h:            float = 0.0
    l_v:            float = 0.0

    # Tail aspect ratios (used to derive b_h, b_v from areas)
    AR_h:           float = 4.5
    AR_v:           float = 1.7

    # Volume-coefficient targets
    V_h_target:     float = 1.00
    V_v_target:     float = 0.10

    V_h_tol:        float = 0.10
    V_v_tol:        float = 0.10

    # Mass and speeds
    MTOW:           float = 0.0
    V_stall:        float = 0.0
    V_cruise:       float = 0.0

    # OEI / rudder
    T_TO:           float = 0.0           # Per-engine thrust at V_MC [N]
    N_propellers:      int   = 0
    y_engine_2:       float = 0.0
    y_engine_4:       float = 0.0
    d_propfan:       float = 0.0
    d_fuselage:      float = 0.0
    V_MC_factor:    float = 1.13
    delta_r_max:    float = np.deg2rad(35)

    # Aileron / roll
    phi_req:        float = 30.0
    t_roll:         float = 7.0
    C_l_p:          float = -0.45
    delta_a_max:    float = np.deg2rad(20)
    eta_i:          float = 0.80
    eta_o:          float = 0.90
    tau_a:          float = 0.42

    # Surface-area fraction limits, empirical data
    Se_Sh_min:      float = 0.22
    Se_Sh_max:      float = 0.40
    Sr_Sv_min:      float = 0.25
    Sr_Sv_max:      float = 0.35

    # Added variables for horizontal tail sizing
    M_landing:      float = 0.0
    G:              float = 9.80665
    rho_ground:     float = 1.225

    # fraction from scissor plot
    S_h_frn:        float = 0.0

    @classmethod
    def from_config(
        cls,
        cfg:   AircraftConfig,
        MTOW:  Optional[float] = None,
        S_ref: Optional[float] = None,
        b:     Optional[float] = None,
        MAC:   Optional[float] = None,
        M_landing: Optional[float] = None,
    ) -> "TailSizing_Input":
        return cls(
            S_ref       = S_ref if S_ref is not None else cfg.S_ref,
            MAC         = MAC   if MAC   is not None else cfg.MAC,
            b           = b     if b     is not None else cfg.b,
            AR          = cfg.AR,
            l_h         = cfg.l_h,
            l_v         = cfg.l_v,
            AR_h        = cfg.AR_h,
            AR_v        = cfg.AR_v,
            V_h_target  = cfg.V_h_target,
            V_v_target  = cfg.V_v_target,
            MTOW        = MTOW if MTOW is not None else cfg.MTOW_initial,
            M_landing   = M_landing if M_landing is not None else (MTOW if MTOW is not None else cfg.MTOW_initial), 
            V_stall     = cfg.V_stall,
            V_cruise    = cfg.V_cruise,
            T_TO        = cfg.T_TO_per_engine,
            y_engine_2    = cfg.y_engine_2,
            y_engine_4  =cfg.y_engine_4,
            d_propfan = cfg.D_propfan,
            N_propellers   = cfg.N_propellers,
            d_fuselage       = cfg.d_f,
            S_h_frn         = cfg.S_h_frn
        )


@dataclass
class TailSizingBreakdown:

    # Final sized areas
    S_h:            float = 0.0
    S_v:            float = 0.0
    b_h:            float = 0.0
    b_v:            float = 0.0

    # HT sizing components
    S_h_stability:  float = 0.0       # from V_h_target
    S_h_control:    float = 0.0       # from elevator power (approx)
    S_h_driver:     str   = ""

    # VT sizing components
    S_v_stability:  float = 0.0       # from V_v_target
    S_v_control:    float = 0.0       # from OEI rudder requirement
    S_v_driver:     str   = ""

    # Achieved volume coefficients
    V_h:            float = 0.0
    V_v:            float = 0.0
    V_h_ok:         bool  = False
    V_v_ok:         bool  = False

    # Elevator
    S_elevator:     float = 0.0
    Se_Sh:          float = 0.0

    # Rudder
    S_rudder:       float = 0.0
    Sr_Sv:          float = 0.0
    M_engine:       float = 0.0
    M_rudder:       float = 0.0
    OEI_ok:         bool  = False

    # Aileron
    S_aileron:      float = 0.0
    Sa_Sref:        float = 0.0
    p_achieved:     float = 0.0
    p_required:     float = 0.0
    roll_ok:        bool  = False

    def summary(self):
        def get_status(b):
            return Text("OK", style="bold green") if b else Text("FAIL", style="bold red")

        # Horizontal Tail Table
        ht_table = Table(title=f"Horizontal Tail (Driver: {self.S_h_driver})", show_header=True)
        ht_table.add_column("Parameter")
        ht_table.add_column("Area (m^2)", justify="right")
        ht_table.add_column("V_h / Ratio", justify="right")
        ht_table.add_row("Stability Requisite", f"{self.S_h_stability:.2f}", f"Target: 1.00")
        ht_table.add_row("Control Requisite", f"{self.S_h_control:.2f}", f"S_e/S_h: {self.Se_Sh:.3f}")
        ht_table.add_row("[bold]Final S_h[/bold]", f"[bold cyan]{self.S_h:.2f}[/bold cyan]", f"Achieved V_h: {self.V_h:.3f}", end_section=True)
        ht_table.add_row("Status", "", get_status(self.V_h_ok))

        # Vertical Tail Table
        vt_table = Table(title=f"Vertical Tail (Driver: {self.S_v_driver})", show_header=True)
        vt_table.add_column("Parameter")
        vt_table.add_column("Value", justify="right")
        vt_table.add_column("Status", justify="center")
        vt_table.add_row("Stability (V_v)", f"S_v = {self.S_v_stability:.2f} m^2", get_status(self.V_v_ok))
        vt_table.add_row("OEI Rudder Moment", f"{self.M_rudder/1000:.1f} kNm", get_status(self.OEI_ok))
        vt_table.add_row("[bold]Final S_v[/bold]", f"[bold cyan]{self.S_v:.2f} m^2[/bold cyan]", "")

        # Roll Table
        roll_table = Table(title="Roll Control (Ailerons)", show_header=True)
        roll_table.add_column("Requirement")
        roll_table.add_column("Achieved")
        roll_table.add_column("Status")
        roll_table.add_row(f"{np.rad2deg(self.p_required):.1f}°/s", f"{np.rad2deg(self.p_achieved):.1f}°/s", get_status(self.roll_ok))

        return Group(ht_table, vt_table, roll_table)


class TailSizingEstimator:

    g  = 9.80665
    CL_alpha_v  = 3.5
    tau_r       = 0.6
    k_r = CL_alpha_v * tau_r           # Torenbeek rudder effectiveness
    rho_SL = 1.225

    def __init__(self, inputs: TailSizing_Input):
        self.i = inputs
        self._validate()

    def _validate(self):
        d = self.i
        required = dict(
            S_ref=d.S_ref, MAC=d.MAC, b=d.b,
            l_h=d.l_h, l_v=d.l_v,
            MTOW=d.MTOW, V_stall=d.V_stall, V_cruise=d.V_cruise,
            T_TO=d.T_TO, y_engine_2=d.y_engine_2, y_engine_4=d.y_engine_4,d_propfan=d.d_propfan,d_fuselage=d.d_fuselage
        )
        missing = [k for k, v in required.items() if v <= 0]
        if missing:
            raise ValueError(f"Inputs not set or zero: {missing}")

    # ------------------------------------------------------------------
    # Horizontal tail
    # ------------------------------------------------------------------
    def _S_h_stability(self) -> float:
        d = self.i
        return d.V_h_target * d.S_ref * d.MAC / d.l_h

    def _S_h_control(self) -> float:
        """
        Approximate elevator-power requirement.

        Quick Class II proxy: require enough elevator authority at the
        landing-approach condition to trim a CG range of ~10% MAC.

            dC_M_required ~ 0.05  (CG shift)
            dC_M = -V_h * C_L_alpha_h * tau_e * delta_e_max

        Solving for V_h gives the minimum, then S_h follows.

        Numbers: C_L_alpha_h ~ 4.5 /rad (untwisted), tau_e ~ 0.5
        (elevator effectiveness for c_e/c ~ 0.30), delta_e_max = 25 deg.

        This is much coarser than a proper scissor-plot analysis but
        gives a sanity-check lower bound on S_h.
        """
        d               = self.i
        CL_approach     = 2*d.M_landing*d.G/(d.rho_ground*d.S_ref*(d.V_stall*1.3)**2)
        delta_CG        = 0.10                  # 10% MAC CG range
        dCM_required    = CL_approach * delta_CG
        C_L_alpha_h     = 4.5                   # /rad, reasonable for unswept HT
        tau_e           = 0.50                  # c_e/c ~ 0.30 (Torenbeek Fig 9)
        delta_e_max     = np.deg2rad(25)
        V_h_min         = dCM_required / (C_L_alpha_h * tau_e * delta_e_max)
        return V_h_min * d.S_ref * d.MAC / d.l_h

    def _S_h_scissor(self) -> float:
        d = self.i
        #---------------- value read from scissor plot ------------------------
        S_h_frn = d.S_h_frn
        return S_h_frn * d.S_ref

    # ------------------------------------------------------------------
    # Vertical tail
    # ------------------------------------------------------------------
    def _S_v_stability(self) -> float:
        d = self.i
        return d.V_v_target * d.S_ref * d.b / d.l_v

    def _S_v_control(self) -> float:
        """
        OEI rudder sizing per CS-25.149 / Torenbeek.

        At V_MC = 1.13 V_stall, the working engine produces an asymmetric
        yaw moment M_engine = T_TO * y_engine. The vertical tail with
        rudder must restore this moment at maximum rudder deflection.

            M_rudder = k_r * S_r * l_v * q_mc * delta_r_max
            S_r      = Sr_Sv_max * S_v        (rudder fills the cap)

        Setting M_rudder = M_engine and solving for S_v gives the
        minimum tail area for OEI control.
        """
        d        = self.i
        V_mc     = d.V_MC_factor * d.V_stall
        q_mc     = 0.5 * self.rho_SL * V_mc**2
        q_mc=q_mc
        if d.N_propellers > 2:
            M_engine = d.T_TO *0.8 * (d.y_engine_4+d.d_propfan/2+d.d_fuselage/2)       #since only 80% of thrust is available for worst case scenario with 4 engines, per CS-25.149
        elif d.N_propellers == 2:
            M_engine = d.T_TO * (d.y_engine_2+d.d_propfan/2+d.d_fuselage/2)       

        S_v_min = M_engine / (
            self.k_r * d.Sr_Sv_max * d.l_v * q_mc * d.delta_r_max
        )
        return S_v_min

    # ------------------------------------------------------------------
    # Achieved volume coefficients
    # ------------------------------------------------------------------
    def _check_volume_coefs(self, S_h: float, S_v: float):
        d = self.i
        V_h    = S_h * d.l_h / (d.S_ref * d.MAC)
        V_v    = S_v * d.l_v / (d.S_ref * d.b)
        V_h_ok = V_h >= d.V_h_target * (1 - d.V_h_tol)
        V_v_ok = V_v >= d.V_v_target * (1 - d.V_v_tol)
        return V_h, V_v, V_h_ok, V_v_ok

    # ------------------------------------------------------------------
    # Surface areas
    # ------------------------------------------------------------------
    def _elevator(self, S_h: float):
        d     = self.i
        Se_Sh = (d.Se_Sh_min + d.Se_Sh_max) / 2
        return Se_Sh * S_h, Se_Sh

    def _rudder(self, S_v: float):
        """
        Compute the rudder area at this S_v. With S_v sized for OEI control,
        Sr_Sv hits its max and M_rudder = M_engine by construction.
        """
        d        = self.i
        V_mc     = d.V_MC_factor * d.V_stall
        q_mc     = 0.5 * self.rho_SL * V_mc**2
        if d.N_propellers > 2:
            M_engine = d.T_TO *0.8 * (d.y_engine_4)       #since only 80% of thrust is available for worst case scenario with 4 engines, per CS-25.149
        elif d.N_propellers == 2:
            M_engine = d.T_TO * (d.y_engine_2+d.d_propfan/2+d.d_fuselage/2)       #since only 80% of thrust is available for worst case scenario with 4 engines, per CS-25.149

        S_r_required = M_engine / (self.k_r * d.l_v * q_mc * d.delta_r_max)
        Sr_Sv        = np.clip(S_r_required / S_v, d.Sr_Sv_min, d.Sr_Sv_max)
        S_r          = Sr_Sv * S_v

        M_rudder = self.k_r * S_r * d.l_v * q_mc * d.delta_r_max
        return S_r, Sr_Sv, M_engine, M_rudder, M_rudder >= M_engine - 1.0

    def _ailerons(self):
        d      = self.i
        p_req  = np.deg2rad(d.phi_req) / d.t_roll
        C_l_da = (np.pi / 4) * (d.eta_o**2 - d.eta_i**2) * d.tau_a
        p_achieved = (C_l_da * d.delta_a_max / (-d.C_l_p)) * (2 * d.V_stall*1.2 / d.b)

        c_mean = d.S_ref / d.b
        c_a    = 0.27 * c_mean
        S_a    = 2 * (d.eta_o - d.eta_i) * (d.b / 2) * c_a

        Sa_Sref = S_a / d.S_ref
        return S_a, Sa_Sref, p_achieved, p_req, p_achieved >= p_req

    # ------------------------------------------------------------------
    def compute(self) -> TailSizingBreakdown:
        self._validate()
        d = self.i

        S_h_stab = self._S_h_stability()
        S_h_ctrl = self._S_h_control()
        S_h_scissor = self._S_h_scissor()

        # old statement:
        # if S_h_stab >= S_h_ctrl:
        #     S_h, S_h_drv = S_h_stab, "stability (V_h)"
        # else:
        #     S_h, S_h_drv = S_h_ctrl, "control (elevator)"

        if S_h_stab >= S_h_ctrl and S_h_stab >= S_h_scissor:
            S_h, S_h_drv = S_h_stab, "stability (V_h)"
        elif S_h_ctrl >= S_h_stab and S_h_ctrl >= S_h_scissor:
            S_h, S_h_drv = S_h_ctrl, "control (elevator)"
        elif S_h_scissor >= S_h_stab and S_h_scissor >= S_h_ctrl:
            S_h, S_h_drv = S_h_scissor, "scissor plot"

        S_v_stab = self._S_v_stability()
        S_v_ctrl = self._S_v_control()
        if S_v_stab >= S_v_ctrl:
            S_v, S_v_drv = S_v_stab, "stability (V_v)"
        else:
            S_v, S_v_drv = S_v_ctrl, "control (OEI rudder)"

        V_h, V_v, V_h_ok, V_v_ok            = self._check_volume_coefs(S_h, S_v)
        S_e, Se_Sh                          = self._elevator(S_h)
        S_r, Sr_Sv, M_eng, M_rud, OEI_ok    = self._rudder(S_v)
        S_a, Sa_Sref, p_ach, p_req, rol_ok  = self._ailerons()

        b_h = np.sqrt(d.AR_h * S_h)
        b_v = np.sqrt(d.AR_v * S_v)

        return TailSizingBreakdown(
            S_h=S_h, S_v=S_v, b_h=b_h, b_v=b_v,
            S_h_stability=S_h_stab, S_h_control=S_h_ctrl, S_h_driver=S_h_drv,
            S_v_stability=S_v_stab, S_v_control=S_v_ctrl, S_v_driver=S_v_drv,
            V_h=V_h, V_v=V_v, V_h_ok=V_h_ok, V_v_ok=V_v_ok,
            S_elevator=S_e, Se_Sh=Se_Sh,
            S_rudder=S_r, Sr_Sv=Sr_Sv, M_engine=M_eng, M_rudder=M_rud, OEI_ok=OEI_ok,
            S_aileron=S_a, Sa_Sref=Sa_Sref,
            p_achieved=p_ach, p_required=p_req, roll_ok=rol_ok,
        )


if __name__ == "__main__":
    from Aircraft_Config import default_q400_hycool

    cfg = default_q400_hycool()
    inp = TailSizing_Input.from_config(cfg)
    est = TailSizingEstimator(inp)
    print(est.compute().summary())