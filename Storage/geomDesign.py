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
    h_fill: float
    h_vent: float
    Sw_fill: dict   # wetted area at fill
    Sw_vent: dict   # wetted area at vent

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
        self.vl_vent = props.saturated_specific_volumes(self.p_vent * self.BAR, FLUID)[0] 
        self.vg_vent = props.saturated_specific_volumes(self.p_vent * self.BAR, FLUID)[1]

    # Calculate the liquid mass fraction at a given pressure and initial liquid mass fraction.
    def calculateInitialLiquidMassFraction(self, yl_vent):        
        """Calculate the liquid mass fraction at a given pressure based on the initial liquid mass fraction and the pressure ratio."""
        # Placeholder: replace with actual calculation using properties

        rhol = (yl_vent / self.vl_vent) + ((1 - yl_vent) / self.vg_vent)  # Example calculation; adjust as needed
        yl_0 = (rhol - (1/self.vg_0)) / (1/self.vl_0 - 1/self.vg_0)  # Example calculation; adjust as needed
        return yl_0

    # Calculated the required tank volume based on the mass of hydrogen, initial liquid mass fraction, and density.
    def calculateTankVolume(self, rho_H2, m_H2, yl_0):
        """Calculate the required tank volume V from the input parameters."""
        # Placeholder: replace with actual calculation using properties and geometry
        V_tank = m_H2 / (rho_H2 * yl_0)  # Example calculation; adjust as needed
        return V_tank 
    
    # Calculate the tank geometry (e.g., radius, length) based on the volume and design constraints.
    def calculateTankGeometry(self, V_tank, phi, psi, Lambda, yl_fill):
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
        if Lambda == 0:
            A_heads = 4 * np.pi * (b**2)
        else:
            p = 1.6075  # Parameter for a general spheroid in Thomsens' approximation
            A_heads = 4*np.pi * (((a*c)**p + (a*b)**p + (c*b)**p) / 3)**(1/p)

        # Calculate total surface area
        A_tank = A_body + A_heads

        # Calculate the height of the LH2 at different fill conditions
        h_fill = self.solve_fill_height(R=b, L=ls, yl=yl_fill, V=V_tank)
        h_vent = self.solve_fill_height(R=b, L=ls, yl=self.y_max, V=V_tank)

        # Calculate wetted surface area
        Sw_vent, Sw_fill = self.calculateWettedAreas(R=b, L=ls, h_fill=h_fill, h_vent=h_vent)

        return Geometry(a=a, b=b, c=c, ls=ls, lt=lt, V_tank=V_tank, A_tank=A_tank, h_fill=h_fill, h_vent=h_vent, Sw_vent=Sw_vent, Sw_fill=Sw_fill)
   
    # Calculate the wetted area
    def calculateWettedAreas(self, R, L, h_fill, h_vent):
        """Calculate the wetted areas of the tank surfaces at fill and venting conditions."""

        ##############################################################################

        Sw_l_vent = 2 * L * R * np.arccos(((R - h_vent)/R)) + 2 * np.pi * R * h_vent  # Example calculation; replace with actual calculation
        Sw_g_vent = 2 * np.pi * R * L + 4 * np.pi * (R ** 2) - Sw_l_vent # Example calculation; replace with actual calculation

        Sw_l_floor_vent = 2 * L * R * np.arccos((R - h_vent) / R) # Example calculation; replace with actual calculation
        Sw_l_caps_vent = 2 * np.pi * R * h_vent  # Example calculation; replace with actual calculation
        Sw_g_ceil_vent = 2 * L * R * (np.pi - np.arccos((R - h_vent) / R))  # Example calculation; replace with actual calculation
        Sw_g_caps_vent = 2 * np.pi * R * (2 * R - h_vent)  # Example calculation; replace with actual calculation
        
        ##############################################################################

        Sw_l_fill = 2 * L * R * np.arccos(((R - h_fill)/R)) + 2 * np.pi * R * h_fill # Example calculation; replace with actual calculation
        Sw_g_fill = 2 * np.pi * R * L + 4 * np.pi * (R ** 2) - Sw_l_fill  # Example calculation; replace with actual calculation

        Sw_l_floor_fill = 2 * L * R * np.arccos((R - h_fill) / R)  # Example calculation; replace with actual calculation
        Sw_l_caps_fill = 2 * np.pi * R * h_fill  # Example calculation; replace with actual calculation
        Sw_g_ceil_fill = 2 * L * R * (np.pi - np.arccos((R - h_fill) / R))  # Example calculation; replace with actual calculation
        Sw_g_caps_fill = 2 * np.pi * R * (2 * R - h_fill)  # Example calculation; replace with actual calculation

        ##############################################################################

        Sw_vent = {
            "total": Sw_l_vent + Sw_g_vent,
            "total LH2": Sw_l_vent,
            "total GH2": Sw_g_vent,
            "floor LH2": Sw_l_floor_vent,
            "caps LH2": Sw_l_caps_vent,
            "caps GH2": Sw_g_caps_vent,
            "ceil GH2": Sw_g_ceil_vent
        }

        Sw_fill = {
            "total": Sw_l_fill + Sw_g_fill,
            "total LH2": Sw_l_fill,
            "total GH2": Sw_g_fill,
            "floor LH2": Sw_l_floor_fill,
            "caps LH2": Sw_l_caps_fill,
            "caps GH2": Sw_g_caps_fill,
            "ceil GH2": Sw_g_ceil_fill
        }
        
        return Sw_vent, Sw_fill

    def solve_fill_height(self, R, L, yl, V):
        """
        Solve for the fill height h [m] such that:
            V_liquid(h) = (1 - ullage_frac) * V_total
    
        Uses Brent's method on the bracketed interval (0, 2R).
        """
        V_liq_target = yl * V

        def V_liquid(h, R, L):
            """Total liquid volume at fill height h [m³]."""
            theta = np.arccos((R - h) / R) 
            V_cylinder = L * (R**2 * theta - (R - h) * np.sqrt(2*R*h - h**2))
            V_sphere = (np.pi * h**2 / 3) * (3*R - h)
            V_liq = V_cylinder + V_sphere
            return V_liq
    
        def residual(h):
            return V_liquid(h, R, L) - V_liq_target
    
        # Sanity check: target must be within physical bounds
        V_min = V_liquid(0,   R, L)   # = 0
        V_max = V_liquid(2*R, R, L)   # = V_total
    
        if V_liq_target <= V_min or V_liq_target >= V_max:
            raise ValueError(
                f"Target liquid volume {V_liq_target:.4f} m³ is outside the "
                f"physical range [{V_min:.4f}, {V_max:.4f}] m³."
            )
        
        h_solution = brentq(residual, 1e-9, 2*R - 1e-9, xtol=1e-12, rtol=1e-12)
        return h_solution
        
def print_params(p_vent, p_fill, y_max, yl_0, rho_H2_vent, rho_H2_fill):
    fields = [
        ("p_vent", "Venting pressure",            p_vent,  "bar"     ),
        ("p_fill", "Fill pressure",               p_fill,  "bar"     ),
        ("y_max",  "Max liq. mass frac. at vent", y_max,   "-"       ),
        ("yl_0",   "Initial liq. mass fraction",  yl_0,    "-"       ),
        ("ullage_factor", "Ullage factor at fill", 1 - yl_0, "-"       ),
        ("rho_H2_vent", "LH2 density at venting",      rho_H2_vent,  "kg/m^3"  ),
        ("rho_H2_fill", "LH2 density at fill",      rho_H2_fill,  "kg/m^3"  )
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
    m_H2 = 494.67 # kg, mass of hydrogen to be stored

    # 1. Construct once — __init__ caches the fill-condition specific volumes
    p_fill = 1.0 # bar, fill pressure
    p_vent = 1.25 * p_fill # bar, venting pressure
    geom_design = GeomDesign(p_vent=p_vent, p_fill=p_fill, y_max=0.97)

    # 2. Initial liquid fraction at fill so that y_l = y_max at venting
    yl_0 = geom_design.calculateInitialLiquidMassFraction(yl_vent=geom_design.y_max)

    # 3. Mean mixture density at venting (from CoolProp / fluids module)
    rho_H2_vent = PropsSI('D', 'P', geom_design.p_vent*geom_design.BAR, 'Q', 0, 'Hydrogen')
    rho_H2_fill = PropsSI('D', 'P', geom_design.p_fill*geom_design.BAR, 'Q', 0, 'Hydrogen')

    # 4. Calculate the required volume
    V_tank = geom_design.calculateTankVolume(rho_H2=rho_H2_vent, m_H2=m_H2, yl_0=yl_0)

    # 5. Dimensions for the chosen shape (area comes for free via the property)
    geom = geom_design.calculateTankGeometry(V_tank, phi=1.0, psi=1.0, Lambda=0., yl_fill=yl_0)

    # 6. Print design parameters and geometry summary
    print_params(p_vent=geom_design.p_vent, p_fill=geom_design.p_fill,
                 y_max=geom_design.y_max, yl_0=yl_0, rho_H2_vent=rho_H2_vent, rho_H2_fill=rho_H2_fill)
    geom.print_summary()
    