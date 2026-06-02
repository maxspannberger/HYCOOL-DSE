"""
Sizes the tank and calculates the dimensions of the tank. 
"""

from CoolProp.CoolProp import PropsSI
from scipy.integrate import quad
from scipy.optimize import brentq
import numpy as np
from dataclasses import dataclass
import properties as props

FLUID = "parahydrogen"   # physically correct LH2 near 20 K; 'hydrogen' = normal H2

@dataclass(frozen=True)
class Geometry:
    a: float        # semi-axis (φ·c)
    b: float        # head half-length (ψ·c)
    c: float        # reference semi-axis
    ls: float      # straight shell length
    lt: float      # total length
    V_tank: float      # tank volume
    A_tank: float   # total surface area

    def print_summary(self):
        fields = [
            ("a",      "Semi-major axis (phi*c)",        self.a,       "m"   ),
            ("b",      "Head half-length (psi*c)", self.b,       "m"   ),
            ("c",      "Reference semi-minor axis",      self.c,       "m"   ),
            ("ls",     "Straight shell length",    self.ls,      "m"   ),
            ("lt",     "Total length",             self.lt,      "m"   ),
            ("V_tank", "Tank volume",              self.V_tank,  "m^3" ),
            ("A_tank", "Total surface area",       self.A_tank,  "m^2" ),
        ]
        w = 58
        print("\n" + "=" * w)
        print(f"{'  TANK GEOMETRY SUMMARY':^{w}}")
        print("=" * w)
        print(f"  {'Var':<7}  {'Description':<24}  {'Value':>10}  Unit")
        print("-" * w)
        for var, desc, val, unit in fields:
            print(f"  {var:<7}  {desc:<24}  {val:>10.4f}  {unit}")
        print("=" * w + "\n")

class GeomDesign:
    def __init__(self, p_vent, p_fill, y_max):
        self.p_vent = p_vent
        self.p_fill = p_fill
        self.y_max = y_max

        self.BAR = 100000  # Pa, 1 bar in SI units
        self.vl_0 = props.saturated_specific_volumes(self.p_fill * self.BAR, FLUID)[0]
        self.vg_0 = props.saturated_specific_volumes(self.p_fill * self.BAR, FLUID)[1]

    # Calculate the liquid mass fraction at a given pressure and initial liquid mass fraction.
    def calculateLiquidMassFraction(self, yl_0):        
        """Calculate the liquid mass fraction at a given pressure based on the initial liquid mass fraction and the pressure ratio."""
        # Placeholder: replace with actual calculation using properties
        rho0 = yl_0 / self.vl_0 + (1.0 - yl_0) / self.vg_0  # Example calculation; adjust as needed
        yl = (rho0 - (1/self.vg_0)) / (1/self.vl_0 - 1/self.vg_0)  # Example calculation; adjust as needed
        return yl

    # Calculate the initial liquid mass fraction to achieve a target liquid mass fraction at venting pressure.
    def calculateInitialLiquidMassFraction(self, p_vent, p_fill, y_l_vent=0.97, yl_0_bracket=(0.50, 0.999)):
        p_fill = p_fill * self.BAR  # Convert p_fill from bar to Pa
        p_vent = p_vent * self.BAR  # Convert p_vent from bar to Pa
        if p_vent <= p_fill:
            raise ValueError(
                f"p_vent ({p_vent/self.BAR:.3f} bar) must exceed fill pressure "
                f"p_fill ({p_fill/self.BAR:.3f} bar)."
            )

        # Residual = how far the forward-mapped liquid fraction at P_vent sits
        # from the target. We want this to be zero.
        def residual(yl_0):
            residual_value = self.calculateLiquidMassFraction(yl_0) - y_l_vent
            return residual_value

        lo, hi = yl_0_bracket
        f_lo, f_hi = residual(lo), residual(hi)

        # brentq needs the root bracketed: the residual must change sign across
        # [lo, hi]. If it doesn't, the target isn't reachable in that range.
        if f_lo * f_hi > 0:
            raise ValueError(
                f"Target y_l={y_l_vent} not bracketed by y_l0 in [{lo}, {hi}] "
                f"at P_vent={p_vent/self.BAR:.3f} bar "
                f"(residuals {f_lo:.3g}, {f_hi:.3g}). Widen the bracket."
            )

        yl_0 = brentq(residual, lo, hi, xtol=1e-10, rtol=1e-12)
        return yl_0

    # Calculated the required tank volume based on the mass of hydrogen, initial liquid mass fraction, and density.
    def calculateTankVolume(self, rho_H2, m_H2, yl_0):
        """Calculate the required tank volume V from the input parameters."""
        # Placeholder: replace with actual calculation using properties and geometry
        V_tank = m_H2 / (rho_H2 * yl_0)  # Example calculation; adjust as needed
        return V_tank
    
    # Calculate the tank geometry (e.g., radius, length) based on the volume and design constraints.
    def calculateTankGeometry(self, V_tank, phi, psi, Lambda):
        """Calculate the tank geometry (e.g., radius, length) based on the volume and design constraints."""
        # Placeholder: replace with actual geometric calculations
        c = ( V_tank / (np.pi * phi * psi * ( (2 * Lambda / (1 - Lambda)) + 4/3)) ) ** (1/3)  # e.g., semi-minor axis of ellipsoid
        a = phi * c  # e.g., semi-major axis of ellipsoid
        b = psi * c  # e.g., cap length 
        ls = 2 * b * Lambda / (1 - Lambda) # e.g., length of straight section
        lt = ls + 2 * b # e.g., total length

        # Calculate the surface area of an elliptical shell using Ramanujan's approximation for the perimeter of an ellipse
        h = ((a - c) / (a + c))**2
        p_ellipse = np.pi * (a + c) * (1 + 3*h / (10 + np.sqrt(4 - 3*h)))
        A_body = p_ellipse * ls

        # Calculate the surface area of an spheroid using Thomsens' approximation
        p = 1.6075  # Parameter for a general spheroid in Thomsens' approximation
        A_heads = 4*np.pi * (((a*c)**p + (a*b)**p + (c*b)**p) / 3)**(1/p)
    
        # Calculate total surface area
        A_tank = A_body + A_heads

        return Geometry(a=a, b=b, c=c, ls=ls, lt=lt, V_tank=V_tank, A_tank=A_tank)
    
    
def print_params(p_vent, p_fill, y_max, yl_0, rho_H2_vent, rho_H2_fill):
    fields = [
        ("p_vent", "Venting pressure",            p_vent,  "bar"     ),
        ("p_fill", "Fill pressure",               p_fill,  "bar"     ),
        ("y_max",  "Max liq. mass frac. at vent", y_max,   "-"       ),
        ("yl_0",   "Initial liq. mass fraction",  yl_0,    "-"       ),
        ("rho_H2_vent", "LH2 density at venting",      rho_H2_vent,  "kg/m^3"  ),
        ("rho_H2_fill", "LH2 density at fill",      rho_H2_fill,  "kg/m^3"  ),
    ]
    w = 58
    print("\n" + "=" * w)
    print(f"{'  DESIGN PARAMETERS':^{w}}")
    print("=" * w)
    print(f"  {'Var':<7}  {'Description':<24}  {'Value':>10}  Unit")
    print("-" * w)
    for var, desc, val, unit in fields:
        print(f"  {var:<7}  {desc:<24}  {val:>10.4f}  {unit}")
    print("=" * w + "\n")


if __name__ == "__main__":
    m_H2 = 250 # kg, mass of hydrogen to be stored

    # 1. Construct once — __init__ caches the fill-condition specific volumes
    geom_design = GeomDesign(p_vent=2.0, p_fill=1.2, y_max=0.97)

    # 2. Initial liquid fraction at fill so that y_l = y_max at venting
    yl_0 = geom_design.calculateInitialLiquidMassFraction(
        p_vent=geom_design.p_vent, p_fill=geom_design.p_fill, y_l_vent=geom_design.y_max
    )

    # 3. Mean mixture density at venting (from CoolProp / fluids module)
    rho_H2_vent = PropsSI('D', 'P', geom_design.p_vent*geom_design.BAR, 'Q', 0, 'Hydrogen')
    rho_H2_fill = PropsSI('D', 'P', geom_design.p_fill*geom_design.BAR, 'Q', 0, 'Hydrogen')
    # 4. Calculate the required volume
    V_tank = geom_design.calculateTankVolume(rho_H2=rho_H2_vent, m_H2=m_H2, yl_0=yl_0)

    # 5. Dimensions for the chosen shape (area comes for free via the property)
    geom = geom_design.calculateTankGeometry(V_tank, phi=1.0, psi=1.0, Lambda=0.)

    # 6. Print design parameters and geometry summary
    print_params(p_vent=geom_design.p_vent, p_fill=geom_design.p_fill,
                 y_max=geom_design.y_max, yl_0=yl_0, rho_H2_vent=rho_H2_vent, rho_H2_fill=rho_H2_fill)
    geom.print_summary()
