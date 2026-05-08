import numpy as np
from math import cos
from dataclasses import dataclass, field
from typing import Optional
from ISA import isa
from Aircraft_Config import AircraftConfig

# Physics unchanged from your original. Only addition is from_config(),
# which builds the input from a shared AircraftConfig and accepts optional
# tail-area overrides from the tail-sizing module.
#
# Wetted-area conventions matching your original:
#   S_wet_w = 2 * 1.02 * (S_ref - b_f * c_root / 2)
#   S_wet_h = 2 * 1.02 * (S_h - d_f * MAC_h / 2)
#   S_wet_v = 2 * 1.02 * (S_v - d_f * MAC_v / 4)
# These are reapplied automatically when S_h or S_v change.


@dataclass
class ClassII_Drag_Input:
    # Reference
    S_ref:          float = 0.0

    # Wing
    tc:             float = 0.0
    lambda_half:    float = 0.0
    lambda_tc:      float = 0.0
    MAC:            float = 0.0
    AR:             float = 0.0
    S_wet_w:        float = 0.0
    K_A:            float = 0.87
    e_theo:         float = 0.93

    # Horizontal tail
    tc_h:           float = 0.0
    lambda_h:       float = 0.0
    lambda_tc_h:    float = 0.0
    MAC_h:          float = 0.0
    S_wet_h:        float = 0.0

    # Vertical tail
    tc_v:           float = 0.0
    lambda_v:       float = 0.0
    lambda_tc_v:    float = 0.0
    MAC_v:          float = 0.0
    S_wet_v:        float = 0.0

    # Fuselage
    l_f:            float = 0.0
    d_f:            float = 0.0
    S_wet_f:        float = 0.0

    # Flight condition
    altitude:       float = 0.0
    M_cruise:       float = 0.0
    W_cruise:       float = 0.0

    # Misc
    CD_misc:        float = 0.0003

    @classmethod
    def from_config(
        cls,
        cfg: AircraftConfig,
        MTOW:    Optional[float] = None,
        S_h:     Optional[float] = None,
        S_v:     Optional[float] = None,
        K_A:     float = 0.935,
        e_theo:  float = 0.93,
    ) -> "ClassII_Drag_Input":
        """
        Build drag input from shared config. Tail areas default to the
        config's initial guesses but should be overridden by tail-sizing
        outputs once available.

        W_cruise is the standard mid-cruise approximation 0.95 * MTOW * g.
        """
        MTOW_use = MTOW if MTOW is not None else cfg.MTOW_initial
        S_h_use  = S_h  if S_h  is not None else cfg.S_h_initial
        S_v_use  = S_v  if S_v  is not None else cfg.S_v_initial

        # Wetted areas (same formulas as your original main):
        S_wet_w = 2 * 1.02 * (cfg.S_ref - cfg.b_f * cfg.c_root / 2.0)
        S_wet_h = 2 * 1.02 * (S_h_use   - cfg.d_f * cfg.MAC_h   / 2.0)
        S_wet_v = 2 * 1.02 * (S_v_use   - cfg.d_f * cfg.MAC_v   / 4.0)

        return cls(
            S_ref        = cfg.S_ref,
            tc           = cfg.tc_mean,
            lambda_half  = cfg.sweep_half,
            lambda_tc    = cfg.sweep_tc,
            MAC          = cfg.MAC,
            AR           = cfg.AR,
            S_wet_w      = S_wet_w,
            K_A          = K_A,
            e_theo       = e_theo,

            tc_h         = cfg.tc_h,
            lambda_h     = cfg.sweep_h_half,
            lambda_tc_h  = cfg.sweep_h_tc,
            MAC_h        = cfg.MAC_h,
            S_wet_h      = S_wet_h,

            tc_v         = cfg.tc_v,
            lambda_v     = cfg.sweep_v_half,
            lambda_tc_v  = cfg.sweep_v_tc,
            MAC_v        = cfg.MAC_v,
            S_wet_v      = S_wet_v,

            l_f          = cfg.l_f,
            d_f          = cfg.d_f,
            S_wet_f      = cfg.S_wet_f,

            altitude     = cfg.altitude_cruise,
            M_cruise     = cfg.M_cruise,
            W_cruise     = 0.95 * MTOW_use * 9.80665,
        )


@dataclass
class DragBreakdown:
    CD0_wing:   float = 0.0
    CD0_htail:  float = 0.0
    CD0_vtail:  float = 0.0
    CD0_fus:    float = 0.0
    CD_misc:    float = 0.0

    CD_i:       float = 0.0
    CD_wL:      float = 0.0
    CD_wave:    float = 0.0

    e:          float = 0.0
    CL_cruise:  float = 0.0   # exposed for Breguet / L/D

    @property
    def CD0(self) -> float:
        return self.CD0_wing + self.CD0_htail + self.CD0_vtail + self.CD0_fus + self.CD_misc

    @property
    def CD_lift_dep(self) -> float:
        return self.CD_i + self.CD_wL

    @property
    def CD_total(self) -> float:
        return self.CD0 + self.CD_lift_dep + self.CD_wave

    @property
    def L_over_D(self) -> float:
        if self.CD_total <= 0:
            return 0.0
        return self.CL_cruise / self.CD_total

    def summary(self) -> str:
        lines = [
            "=" * 52,
            "  Class II Drag Breakdown  (Torenbeek, metric)",
            "=" * 52,
            f"  CD0  wing           {self.CD0_wing:.5f}  ({self.CD0_wing*1e4:.1f} cts)",
            f"  CD0  h-tail         {self.CD0_htail:.5f}  ({self.CD0_htail*1e4:.1f} cts)",
            f"  CD0  v-tail         {self.CD0_vtail:.5f}  ({self.CD0_vtail*1e4:.1f} cts)",
            f"  CD0  fuselage       {self.CD0_fus:.5f}  ({self.CD0_fus*1e4:.1f} cts)",
            f"  CD0  misc           {self.CD_misc:.5f}  ({self.CD_misc*1e4:.1f} cts)",
            "-" * 52,
            f"  CD0  total          {self.CD0:.5f}  ({self.CD0*1e4:.1f} cts)",
            f"  CD_i (induced)      {self.CD_i:.5f}  ({self.CD_i*1e4:.1f} cts)",
            f"  CD_wL (lift wave)   {self.CD_wL:.5f}  ({self.CD_wL*1e4:.1f} cts)",
            f"  CD_wave (0-lift)    {self.CD_wave:.5f}  ({self.CD_wave*1e4:.1f} cts)",
            "=" * 52,
            f"  CD_total            {self.CD_total:.5f}  ({self.CD_total*1e4:.1f} cts)",
            f"  CL  cruise          {self.CL_cruise:.4f}",
            f"  L/D cruise          {self.L_over_D:.2f}",
            f"  Oswald e            {self.e:.4f}",
            "=" * 52,
        ]
        return "\n".join(lines)


class DragEstimation:

    IF_wing     = 1.00
    IF_tail     = 1.04
    IF_fuselage = 1.00

    def __init__(self, inputs: ClassII_Drag_Input):
        self.i = inputs
        self._validate()

    def _validate(self):
        d = self.i
        required = dict(
            S_ref=d.S_ref, tc=d.tc, MAC=d.MAC, AR=d.AR,
            S_wet_w=d.S_wet_w, tc_h=d.tc_h, MAC_h=d.MAC_h, S_wet_h=d.S_wet_h,
            tc_v=d.tc_v, MAC_v=d.MAC_v, S_wet_v=d.S_wet_v,
            l_f=d.l_f, d_f=d.d_f, S_wet_f=d.S_wet_f,
            altitude=d.altitude, M_cruise=d.M_cruise, W_cruise=d.W_cruise,
        )
        missing = [k for k, v in required.items() if v <= 0]
        if missing:
            raise ValueError(f"Inputs not set or zero: {missing}")

    def _flight_conditions(self) -> tuple[float, float, float]:
        d           = self.i
        T, p, rho   = isa(d.altitude)
        a           = np.sqrt(1.4 * 287.05 * T)
        V           = d.M_cruise * a
        mu          = 1.716e-5 * (T / 273.15)**1.5 * (273.15 + 110.4) / (T + 110.4)
        q           = 0.5 * rho * V**2
        Re_per_m    = rho * V / mu
        CL          = d.W_cruise / (q * d.S_ref)
        return q, Re_per_m, CL

    def _CF(self, Re: float, lam_frac: float = 0.15) -> float:
        M       = self.i.M_cruise
        CF_turb = 0.455 / (np.log10(Re)**2.58) / ((1 + 0.144*M**2)**0.65)
        CF_lam  = 1.328 / np.sqrt(Re)
        return lam_frac * CF_lam + (1 - lam_frac) * CF_turb

    def _FF_lifting_surf(self, tc: float, lambda_tc: float) -> float:
        return 1 + 2.7 * tc * np.cos(lambda_tc)**2 + 100 * tc**4

    def _FF_fuselage(self) -> float:
        sigma = self.i.l_f / self.i.d_f
        return 1 + 2.2 / sigma**1.5 + 3.8 / sigma**3

    def _cd0_component(self, Re_per_m: float, chord: float,
                       FF: float, IF: float, S_wet: float) -> float:
        Re = Re_per_m * chord
        CF = self._CF(Re)
        return CF * FF * IF * S_wet / self.i.S_ref

    def _M_dd(self, CL: float) -> float:
        d = self.i
        c = np.cos(d.lambda_half)
        return d.K_A / c - d.tc / c**2 - CL / (10 * c**3)

    def _CD_wave(self, CL: float) -> float:
        M   = self.i.M_cruise
        Mdd = self._M_dd(CL)
        if M <= Mdd:
            return 0.0
        return 20 * (M - Mdd)**4

    def _oswald(self, CD0_wing: float) -> float:
        d = self.i
        return 1.0 / (1.0 / d.e_theo + np.pi * d.AR * CD0_wing)

    def _CD_induced(self, CL: float, e: float) -> float:
        return CL**2 / (np.pi * self.i.AR * e)

    def _CD_wave_lift(self, CL: float) -> float:
        M      = self.i.M_cruise
        M_crit = self._M_dd(CL) - 0.04
        if M <= M_crit:
            return 0.0
        return CL**2 * (M - M_crit) * 0.25

    def compute(self) -> DragBreakdown:
        self._validate()
        d = self.i
        q, Re_per_m, CL = self._flight_conditions()

        CD0_w = self._cd0_component(
            Re_per_m, d.MAC,
            self._FF_lifting_surf(d.tc, d.lambda_tc),
            self.IF_wing, d.S_wet_w,
        )
        CD0_h = self._cd0_component(
            Re_per_m, d.MAC_h,
            self._FF_lifting_surf(d.tc_h, d.lambda_tc_h),
            self.IF_tail, d.S_wet_h,
        )
        CD0_v = self._cd0_component(
            Re_per_m, d.MAC_v,
            self._FF_lifting_surf(d.tc_v, d.lambda_tc_v),
            self.IF_tail, d.S_wet_v,
        )
        CD0_f = self._cd0_component(
            Re_per_m, d.l_f,
            self._FF_fuselage(),
            self.IF_fuselage, d.S_wet_f,
        )

        e = self._oswald(CD0_w)

        return DragBreakdown(
            CD0_wing  = CD0_w,
            CD0_htail = CD0_h,
            CD0_vtail = CD0_v,
            CD0_fus   = CD0_f,
            CD_misc   = d.CD_misc,
            CD_i      = self._CD_induced(CL, e),
            CD_wL     = self._CD_wave_lift(CL),
            CD_wave   = self._CD_wave(CL),
            e         = e,
            CL_cruise = CL,
        )


if __name__ == "__main__":
    from Aircraft_Config import default_q400_hycool

    cfg = default_q400_hycool()
    inp = ClassII_Drag_Input.from_config(cfg)
    est = DragEstimation(inp)
    print(est.compute().summary())