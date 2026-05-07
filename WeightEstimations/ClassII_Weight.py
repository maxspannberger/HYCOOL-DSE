import numpy as np
import scipy as sp
from dataclasses import dataclass
from typing import Optional
import Tail_Interpolation as Tail_Interp


# 1) Airframe Structuring
# -> Wings                                  Yes | Formula written
# -> Tail                                   Yes | Formula written
# -> Body                                   Yes | Formula written
# -> Landing gear                           Yes | Formula written
# -> Surface Controls                       Yes | Formula written
# -> Engine Nacelle                         N/A | Propfan / Open fan doesn't have a nacelle

# 2) Propulsion Group
# -> Engine Weight                          Yes | FLOPS / Other

# 3) Airframe Services & Equipment
# -> APU                                    N/A
# -> Instruments, Navigation, Electronics   N/A
# -> Hydraulics & Pneumatics                N/A
# -> Furnishing                             N/A
# -> Air Conditioning and anti-icing        N/A


# Aiframe structure
# W_structure = 0.447 * MTOW * sqrt(N_ult) * (bf * hf * lf / MTOW)^0.24
# N_ult = ultimate load factor (2.5), bf, hf, lf = fuselage dimensions (Width, height, length)

# W_structure = wing + tail + body + Landing gear + Surface control + Nacelle / Engine 

# W_HorizontalStabiliser = 1 * (S_h**0.2 * V_dive / sqrt(cos(Lambda_h)))
# W_VerticalTail = k_v * (S_v**0.2 * V_dive / sqrt(cos(Lambda_v)))
# Where k_v depends on whether it's a t-tail or normal tail.
# For normal, k_v = 1. For t-tail, k_v = 1 + 0.15 * S_h * h_h / (S_v * b_v)

# W_fuselage = k_wf * sqrt(V_dive * l_t / (b_f + h_f)) * S_f_wet**1.2
# k_wf = 0.23-W_f (??), l_t = distance between quarter chord of wing and horizontal tail, h_f = maximum deptjh of fuselage
# S_f_wet = pi * b_f * l_f * (1 - 2 / sigma)**2/3 * (1 + 1 / sigma**2)
# Sigma = fuselage fineness ratio (???)

# W_LG = k_LG * (A + B * W_TO **0.75 + C*W_TO + D*W_TO **1.5)
# k_LG = 1 for Low wing, 1.08 for High wing. A, B, C, D to be found on page 283 of Torenbeek

# W_SC = 1.2 * k_SC * W_TO**2/3
# k_SC =  0.472 (No flap/slat control), 0.567 (Flap/Slat control)


@dataclass
class ClassII_Input:
    # Weights
    MTOW:       float = 0.0         # kg
    MZFW:       float = 0.0         # kg
    n_ult:      float = 3.75        # CS-25 Requirements

    # Wing
    b:          float = 0.0         # m
    S_w:        float = 0.0         # m^2
    sweep_half: float = 0.0         # Radians
    t_r:        float = 0.0         # m
    k_w:        float = 6.67 * 10 ** -3

    # Horizontal tail
    S_h:        float = 0.0         # m^2
    sweep_h:    float = 0.0         # Radians
 
    # Vertical tail
    S_v:        float = 0.0         # m^2
    sweep_v:    float = 0.0         # Radians
    b_v:        float = 0.0         # m
    t_tail:     bool  = False       
    h_h:        float = 0.0         # m
    k_v:        float = 0.0         # Depends on whether it's a T-tail or not
 
    # Fuselage
    b_f:        float = 0.0         # m
    h_f:        float = 0.0         # m
    l_f:        float = 0.0         # m
    l_t:        float = 0.0         # m
    k_wf:       float = 0.23        # Torenbeek
 
    # Speed
    V_dive:     float = 0.0         # m/s

    # Landing gear
    high_wing:  bool = False
 
    # Surface controls
    has_flap_slat: bool = True      # Changes value of wing constant
 
    # Propulsion
    W_Propulsion: float = 0.0       # Define based on your propulsion architecture. Includes power generation & motors


@dataclass
class WeightBreakdown:
    W_wing:   float = 0.0
    W_htail:  float = 0.0
    W_vtail:  float = 0.0
    W_fus:    float = 0.0
    W_lg:     float = 0.0
    W_sc:     float = 0.0
    W_engine: float = 0.0
 
    @property
    def W_structure(self) -> float:
        """Airframe structural weight (excl. propulsion)."""
        return (self.W_wing + self.W_htail + self.W_vtail
                + self.W_fus + self.W_lg + self.W_sc)
 
    @property
    def W_empty(self) -> float:
        """Structural OEW estimate: structure + propulsion group."""
        return self.W_structure + self.W_engine
 
    def summary(self) -> str:
        lines = [
            "=" * 52,
            "  Class II Weight Breakdown  (Torenbeek, metric)",
            "=" * 52,
            f"  Wing              {self.W_wing:>10.1f} kg",
            f"  Horizontal tail   {self.W_htail:>10.1f} kg",
            f"  Vertical tail     {self.W_vtail:>10.1f} kg",
            f"  Fuselage          {self.W_fus:>10.1f} kg",
            f"  Landing gear      {self.W_lg:>10.1f} kg",
            f"  Surface controls  {self.W_sc:>10.1f} kg",
            "-" * 52,
            f"  W_structure       {self.W_structure:>10.1f} kg",
            f"  Propulsion        {self.W_engine:>10.1f} kg",
            "=" * 52,
            f"  W_empty (OEW)     {self.W_empty:>10.1f} kg",
            "=" * 52,
        ]
        return "\n".join(lines)
 

@dataclass
class weightEstimation:

    b_ref           = 1.905         # Torenbeek ref. span

    _LG_main = dict(A = 18.1, B = 0.131, C = 0.019, D = 2.23 * 10 **(-5))
    _LG_nose = dict(A = 9.1,  B = 0.082, C = 0,     D = 2.97 * 10 **(-6))

    def __init__(self, geometry: ClassII_Input):
        self.g = geometry

    def _validate(self):
        g = self.g

        required = dict(
            MTOW=g.MTOW, MZFW=g.MZFW, b=g.b, S_w=g.S_w, t_r=g.t_r,
            S_h=g.S_h, S_v=g.S_v, b_v=g.b_v,
            b_f=g.b_f, h_f=g.h_f, l_f=g.l_f, l_t=g.l_t,
            V_dive=g.V_dive,
        )

        missing = [k for k, v in required.items() if v <= 0]
        if missing:
            raise ValueError(f"Inputs not set or zero: {missing}")
        
        if g.MZFW > g.MTOW:
            raise ValueError("Why is your MZFW bigger than MTOW?.")    
        

    # W_wing = W_G * k_w * b_s**0.75 * (1 + sqrt(b_ref / b_s)) * n_ult**0.55 * (b_s/t_r / W_G/S)**0.3 * 1.02
    # k_w = 6.67E-3 for MTOW > 12500lbs, b_ref = 1.905m, b_s = b * cos(Half sweep angle), t_r = root thickness, W_G = Gross shell weight
    # Torenbeek Eq. (8-12)

    def _wing_weight(self) -> float:
        g       = self.g
        b_s     = g.b * np.cos(g.sweep_half)        # Projected Span

        return g.MZFW * g.k_w * b_s**0.75 * (1 + np.sqrt(self.b_ref / b_s)) * g.n_ult**0.55 * ( (b_s / g.t_r) / (g.MZFW / g.S_w ) )**0.3 * 1.02
    

    # S in m^2 -> ft^2: multiply by 10.7639
    # V in m/s (EAS) -> knots: multiply by 1.94384

    def _tail_x_imperial(S_m2: float, V_mps: float, sweep_rad: float) -> float:
        S_ft2 = S_m2 * 10.7639
        V_kt  = V_mps * 1.94384
        return S_ft2**0.2 * V_kt / 1000 / np.sqrt(np.cos(sweep_rad))

    # W_HorizontalStabiliser = 1 * f(S_h**0.2 * V_dive / sqrt(cos(Lambda_h)))
    # Torenbeek Eq. (8-14)

    def _htail_weight(self) -> float:
        g = self.g
        S_ft2 = g.S_h * 10.7639
        V_kt  = g.V_dive * 1.94384
        x     = S_ft2**0.2 * V_kt / 1000 / np.sqrt(np.cos(g.sweep_h))
        w_per_area_lb_ft2 = Tail_Interp.get_weight_factor(x)   # lb/ft²
        return w_per_area_lb_ft2 * S_ft2 * 0.453592            # -> kg
        
    # W_VerticalTail = k_v * (S_v**0.2 * V_dive / sqrt(cos(Lambda_v)))
    # For normal, k_v = 1. For t-tail, k_v = 1 + 0.15 * S_h * h_h / (S_v * b_v)
    # Torenbeek Eq. (8-15)

    def _vtail_weight(self) -> float:
        g = self.g
        S_ft2 = g.S_v * 10.7639
        V_kt  = g.V_dive * 1.94384
        x     = S_ft2**0.2 * V_kt / 1000 / np.sqrt(np.cos(g.sweep_v))
        w_per_area_lb_ft2 = Tail_Interp.get_weight_factor(x)

        if g.t_tail:
            k_v = 1 + 0.15 * g.S_h * g.h_h / (g.S_v * g.b_v)
        else:
            k_v = 1.0

        return w_per_area_lb_ft2 * S_ft2 * k_v * 0.453592      # -> kg

    # W_fuselage = k_wf * sqrt(V_dive * l_t / (b_f + h_f)) * S_f_wet**1.2
    # k_wf = 0.23-W_f (??), l_t = distance between quarter chord of wing and horizontal tail, h_f = maximum depth of fuselage
    # S_f_wet = pi * b_f * l_f * (1 - 2 / sigma)**2/3 * (1 + 1 / sigma**2)
    # Sigma = fuselage fineness ratio (???)
    # Torenbeek Eq. (8-16)

    def _fuselage_weight(self) -> float:
        g = self.g
        d_eq  = (g.b_f + g.h_f) / 2.0
        sigma = g.l_f / d_eq
        S_f_wet = (np.pi * g.b_f * g.l_f * (1.0 - 2.0 / sigma)**(2.0 / 3.0) * (1.0 + 1.0 / sigma**2))

        return g.k_wf * np.sqrt(g.V_dive * g.l_t / (g.b_f + g.h_f)) * S_f_wet ** 1.2
    
    # W_LG = k_LG * (A + B * W_TO **0.75 + C*W_TO + D*W_TO **1.5)
    # k_LG = 1 for Low wing, 1.08 for High wing. A, B, C, D to be found on page 283 of Torenbeek    

    def _LDG_weight(self) -> float:
        g = self.g
        k_LG = 1.08 if g.high_wing else 1

        def _leg(c: dict) -> float:
            return (c["A"]
                    + c["B"] * g.MTOW**0.75
                    + c["C"] * g.MTOW
                    + c["D"] * g.MTOW**1.5)
 
        return k_LG * (_leg(self._LG_main) + _leg(self._LG_nose)) 
    
    # W_SC = 1.2 * k_SC * W_TO**2/3
    # k_SC =  0.472 (No flap/slat control), 0.567 (Flap/Slat control)

    def _surface_control_weight(self) -> float:
        g = self.g
        k_SC = 0.567 if g.has_flap_slat else 0.472
        return 1.2 * k_SC * g.MTOW ** (2/3)
    
    def _propulsion_weight(self) -> float:
        g = self.g
        return g.W_Propulsion
    
    def compute(self) -> WeightBreakdown:
        self._validate()

        return WeightBreakdown(
            W_wing   = self._wing_weight(),
            W_htail  = self._htail_weight(),
            W_vtail  = self._vtail_weight(),
            W_fus    = self._fuselage_weight(),
            W_lg     = self._LDG_weight(),
            W_sc     = self._surface_control_weight(),
            W_engine = self._propulsion_weight(),
        )
    

    def iterate_MTOW(
        self,
        W_payload:         float,
        W_fuel:            float,
        W_fixed_equipment: float = 0.0,
        tol:               float = 1.0,
        max_iter:          int   = 50,
    ) -> tuple[float, WeightBreakdown]:
        
        for i in range(max_iter):
            bd = self.compute()
            MTOW_new = bd.W_empty + W_payload + W_fuel + W_fixed_equipment
            delta    = abs(MTOW_new - self.g.MTOW)
            self.g.MTOW = MTOW_new
            self.g.MZFW = MTOW_new - W_fuel

            if delta < tol:
                print(f"Converged in {i + 1} iterations.  MTOW = {MTOW_new:.1f} kg")
                return MTOW_new, bd
        print(f"Warning: did not converge after {max_iter} iterations. "
              f"Residual = {delta:.2f} kg")
        return self.g.MTOW, bd
 

if __name__ == "__main__":

    geo = ClassII_Input(
        # Weights (initial guess for iteration)
        MTOW  = 31_237,
        MZFW  = 30_000,
        n_ult = 3.75,
 
        # Wing
        b          = 28.4,
        S_w        = 63.15,
        sweep_half = np.deg2rad(12.5),
        t_r        = 0.18 * 2.54,        # (t/c)_root * c_root  [m]
 
        # Horizontal tail
        S_h     = 13.94, 
        sweep_h = np.deg2rad(10),
 
        # Vertical tail
        S_v     = 14.8,
        sweep_v = np.deg2rad(35.0),
        b_v     = 4.9,
        t_tail  = True,
 
        # Fuselage
        b_f  = 2.69,
        h_f  = 2.80,
        l_f  = 32.8,
        l_t  = 15.5,
        k_wf = 0.23,
 
        # Speed
        V_dive = 213.5,     # m/s EAS
 
        # LG / controls
        high_wing     = True,
        has_flap_slat = True,
 
        # Propulsion
        W_Propulsion  = 2500 # kg

    )

    est = weightEstimation(geo)
 
    print("--- Single-shot ---")
    print(est.compute().summary())
 
    print("\n--- Iterated MTOW ---")
    _, bd = est.iterate_MTOW(W_payload=10_000, W_fuel=600, W_fixed_equipment=5_500)
    print(bd.summary())