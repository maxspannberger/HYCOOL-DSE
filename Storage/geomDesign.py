"""
Sizes the tank and calculates the dimensions of the tank. 
"""

from CoolProp.CoolProp import PropsSI
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
    A_tank: float      # total surface area

    def print_summary(self):
        fields = [
            ("r",      "Tank Radius",        self.a,       "m"   ),
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
    def calculateTankGeometry(self, V_tank, phi, psi, Lambda):
        """Calculate the tank geometry based on the volume and shape parameters."""
        c = (V_tank / (np.pi * phi * psi * ((2 * Lambda / (1 - Lambda)) + 4/3))) ** (1/3)
        a = phi * c
        b = psi * c
        ls = 2 * b * Lambda / (1 - Lambda)
        lt = ls + 2 * b

        # Elliptical shell surface area (Ramanujan's approximation)
        h = ((a - c) / (a + c))**2
        p_ellipse = np.pi * (a + c) * (1 + 3*h / (10 + np.sqrt(4 - 3*h)))
        A_body = p_ellipse * ls

        # End-cap surface area (Thomsen's approximation)
        if Lambda == 0:
            A_heads = 4 * np.pi * b**2
        else:
            p = 1.6075
            A_heads = 4*np.pi * (((a*c)**p + (a*b)**p + (c*b)**p) / 3)**(1/p)

        A_tank = A_body + A_heads

        return Geometry(a=a, b=b, c=c, ls=ls, lt=lt, V_tank=V_tank, A_tank=A_tank)
   


if __name__ == "__main__":
    m_H2 = 494.67 / 2 # kg, mass of hydrogen to be stored

    # 1. Construct once — __init__ caches the fill-condition specific volumes
    p_fill = 1.0 # bar, fill pressure
    p_vent = 1.5 * p_fill # bar, venting pressure
    geom_design = GeomDesign(p_vent=p_vent, p_fill=p_fill, y_max=0.97)

    # 2. Initial liquid fraction at fill so that y_l = y_max at venting
    yl_0 = geom_design.calculateInitialLiquidMassFraction(yl_vent=geom_design.y_max)

    # 3. Mean mixture density at venting (from CoolProp / fluids module)
    
    rho_H2_fill = PropsSI('D', 'P', p_fill*geom_design.BAR, 'T', 20, 'Hydrogen')
    rho_H2_vent = PropsSI('D', 'P', p_vent*geom_design.BAR, 'T', 20, 'Hydrogen')

    # 4. Calculate the required volume
    V_tank = geom_design.calculateTankVolume(rho_H2=rho_H2_vent, m_H2=m_H2, yl_0=yl_0)

    # 5. Dimensions for the chosen shape (area comes for free via the property)
    geom = geom_design.calculateTankGeometry(V_tank, phi=1.0, psi=1.0, Lambda=0.55)

    # 6. Print geometry summary
    geom.print_summary()
    