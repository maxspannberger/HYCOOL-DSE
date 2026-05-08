import numpy as np
from math import cos
from dataclasses import dataclass, field
from ISA import isa

# Drag components:
#   Zero-Lift Drag
#   -> Skin Friction
#   -> Form Drag
#   -> Wave Drag (Shock wave compressibility)
#   Lift-Dependent Drag
#   -> Induced / Vortex Drag
#   -> Wave Drag (Additional from lift at transonic speeds)

# Zero Lift Drag Eqs:
# CD_0 = 1 / S_ref * (CF_i * FF_i * IF_i * S_wet_i) + CD_misc

# CF_i = flat plate skin friction
#      = 0.455 / (log_10 (Re) ^ 2.58) / (1 + 0.144 M^2)^0.65

# FF_i = Form factor drag per component
#   For Wing / Tail:
#       FF = 1 + 2.7 * (t/c) * cos^2(Lambda_tc) + 100 * (t/c)^4
#   For fuselage:
#       FF = 1 + 2.2/sigma^1.5 + 3.8/sigma^3

# IF_i = Interference factor
# IF_i = 1 for Wing
# IF_i = 1.04 for Tail

# CD_misc is miscellaneous drag from stuff like upsweep.
# Torenbeek says between 0.0002-0.0005 for clean transport aircraft

# Wave Drag:
# Only matters if above the mach drag divergence number
# CD_wave = 0 for M < M_dd
# CD_wave = 20 * (M - M_dd)^4   for M > M_dd
# M_dd = K_A / cos(Lambda_half) - (t/c) / cos^2(Lambda_half) - CL / (10 * cos^3(Lambda_half))
# K_A = airfoil technology factor: 0.87 (NACA 6-series), 0.935 (supercritical)

# Lift-Dependent drag Eqs:
#   Vortex Drag:
#       CD_i = CL^2 / (pi * AR * e)
#   e = 1 / (1/e_theo + pi * AR * CD_0_wing)   -- accounts for non-elliptic + viscous
#   e_theo ~ 0.95-0.98 for well-designed wings
# Or a simpler estimate
#   e = 4.61 * (1 - 0.045 * AR^0.68) * cos(Lambda_LE)^0.15 - 3.1

# Lift-dependent wave drag (Eq. 4-83):
# CD_wL = CL^2 * (M - M_crit_L) * f(M)


@dataclass
class ClassII_Drag_Input:
    # --- Reference ---
    S_ref:          float = 0.0     # Wing reference area [m^2]

    # --- Wing ---
    tc:             float = 0.0     # Wing mean t/c [-]  (root+tip)/2
    lambda_half:    float = 0.0     # Half-chord sweep [rad]
    lambda_tc:      float = 0.0     # Sweep at max-thickness line [rad]
                                    # ~= lambda_half for most transport wings
    MAC:            float = 0.0     # Mean aerodynamic chord [m]
    AR:             float = 0.0     # Aspect ratio [-]
    S_wet_w:        float = 0.0     # Wing wetted area [m^2]
                                    # Q400: ~2 * 1.02 * 63.1 = 128.7 m^2
    K_A:            float = 0.87    # 0.87 NACA 6-series, 0.935 supercritical
    e_theo:         float = 0.93    # Theoretical Oswald factor [-]

    # --- Horizontal tail ---
    tc_h:           float = 0.0     # HT mean t/c [-]
    lambda_h:       float = 0.0     # HT half-chord sweep [rad]
    lambda_tc_h:    float = 0.0     # HT sweep at max-thickness line [rad]
    MAC_h:          float = 0.0     # HT mean chord [m]
    S_wet_h:        float = 0.0     # HT wetted area [m^2]  (~2 * S_h)
                                    # Q400: ~2 * 13.94 = 27.9 m^2

    # --- Vertical tail ---
    tc_v:           float = 0.0     # VT mean t/c [-]
    lambda_v:       float = 0.0     # VT half-chord sweep [rad]
    lambda_tc_v:    float = 0.0     # VT sweep at max-thickness line [rad]
    MAC_v:          float = 0.0     # VT mean chord [m]
    S_wet_v:        float = 0.0     # VT wetted area [m^2]  (~2 * S_v)
                                    # Q400: ~2 * 14.8 = 29.6 m^2

    # --- Fuselage ---
    l_f:            float = 0.0     # Fuselage length [m]
    d_f:            float = 0.0     # Equivalent diameter [m] = (b_f + h_f) / 2
    S_wet_f:        float = 0.0     # Fuselage wetted area [m^2]
                                    # Reuse from weight estimation

    # --- Flight condition ---
    altitude:       float = 0.0     # Cruise altitude [m]
    M_cruise:       float = 0.0     # Cruise Mach [-]
    W_cruise:       float = 0.0     # Cruise weight [N]
                                    # Typically ~0.95 * MTOW * g mid-cruise

    # --- Misc ---
    CD_misc:        float = 0.0003  # Torenbeek: 0.0002-0.0005 clean transport


@dataclass
class DragBreakdown:
    # Zero-lift contributions per component
    CD0_wing:   float = 0.0
    CD0_htail:  float = 0.0
    CD0_vtail:  float = 0.0
    CD0_fus:    float = 0.0
    CD_misc:    float = 0.0

    # Lift-dependent
    CD_i:       float = 0.0     # Induced / vortex drag
    CD_wL:      float = 0.0     # Lift-dependent wave drag

    # Compressibility
    CD_wave:    float = 0.0     # Zero-lift wave drag

    # Oswald factor (stored for inspection)
    e:          float = 0.0

    @property
    def CD0(self) -> float:
        """Total zero-lift drag."""
        return self.CD0_wing + self.CD0_htail + self.CD0_vtail + self.CD0_fus + self.CD_misc

    @property
    def CD_lift_dep(self) -> float:
        """Total lift-dependent drag."""
        return self.CD_i + self.CD_wL

    @property
    def CD_total(self) -> float:
        return self.CD0 + self.CD_lift_dep + self.CD_wave

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
            f"  Oswald e            {self.e:.4f}",
            "=" * 52,
        ]
        return "\n".join(lines)


class DragEstimation:

    # Interference factors (fixed by component type)
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
        d = self.i
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
        c  = np.cos(d.lambda_half)
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
        M       = self.i.M_cruise
        M_crit  = self._M_dd(CL) - 0.04    # Torenbeek approximation
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

        e  = self._oswald(CD0_w)

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
        )

if __name__ == "__main__":

    c_root  = 2.54          # m, wing root chord
    b_f     = 2.69          # m, fuselage width
    d_f     = (2.69 + 2.80) / 2    # m, equivalent diameter

    geo = ClassII_Drag_Input(
        S_ref           = 63.1,                                     # Need

        # Wing
        tc              = 0.15,                                     # Need
        lambda_half     = np.deg2rad(20),                           # Need
        lambda_tc       = np.deg2rad(20),                           # Need
        MAC             = 2.49,                                     # Need
        AR              = 12.78,                                    # Need
        S_wet_w         = 2 * 1.02 * (63.1 - b_f * c_root / 2),     # Need
        K_A             = 0.935,                                    # Do not need
        e_theo          = 0.93,                                     # Do not need

        # Horizontal tail
        tc_h            = 0.12,                                     # Need
        lambda_h        = np.deg2rad(10.0),                         # Need
        lambda_tc_h     = np.deg2rad(8.0),                          # Need
        MAC_h           = 2.80,         # corrected: S_h/b_h with AR_h~4.5, Need
        S_wet_h         = 2 * 1.02 * (13.94 - d_f * 2.80 / 2),      # Need

        # Vertical tail
        tc_v            = 0.12,                                     # Need
        lambda_v        = np.deg2rad(35.0),                         # Need      
        lambda_tc_v     = np.deg2rad(33.0),                         # Need
        MAC_v           = 3.02,         # S_v / b_v = 14.8 / 4.9, Need
        S_wet_v         = 2 * 1.02 * (14.8 - d_f * 3.02 / 4),       # Need

        # Fuselage
        l_f             = 32.8,                                     # Need
        d_f             = d_f,                                      # Need
        S_wet_f         = 240.0,                                    # Need

        # Flight condition
        altitude        = 7_620,                                    # Do not need
        M_cruise        = 0.7,                                      # Do not need
        W_cruise        = 0.95 * 28_604 * 9.80665,                  # Need

        CD_misc         = 0.0003,                                   # Do not need
    )

    est = DragEstimation(geo)
    print(est.compute().summary())