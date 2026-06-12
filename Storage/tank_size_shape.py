"""
Sizes the tank and calculates the dimensions of the tank. 
"""

from CoolProp.CoolProp import PropsSI
import numpy as np
from dataclasses import dataclass
import properties as props

FLUID = "parahydrogen"   

@dataclass(frozen=True)
class Geometry:
    a: float        # semi-axis (φ·c)
    b: float        # head half-length (ψ·c)
    c: float        # reference semi-axis
    ls: float      # straight shell length
    lt: float      # total length
    V_tank: float      # tank volume
    A_tank: float      # total surface area
    ull: float

    def print_summary(self):
        fields = [
            ("r",      "Tank Radius",        self.a,       "m"   ),
            ("ls",     "Straight shell length",    self.ls,      "m"   ),
            ("lt",     "Total length",             self.lt,      "m"   ),
            ("V_tank", "Tank volume",              self.V_tank,  "m^3" ),
            ("A_tank", "Total surface area",       self.A_tank,  "m^2" ),
            ("U", "Ullage Factor", self.ull, "-")
        ]

        RESET = '\033[0m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        GREEN = '\033[92m'
        ORANGE = '\033[38;5;208m'
        BOLD = '\033[1m'

        widths = [7, 24, 10, 5]

        def make_border(left, mid, right):
            return left + mid.join('-' * (w + 2) for w in widths) + right

        title_width = sum(widths) + 3 * len(widths) + (len(widths) - 1)
        print(f"\n{BOLD}{'TANK GEOMETRY SUMMARY':^{title_width}}{RESET}")
        print(make_border('+', '+', '+'))
        print('|' + '|'.join(f" {CYAN}{h:^{w}}{RESET} " for h, w in zip(['Var', 'Description', 'Value', 'Unit'], widths)) + '|')
        print(make_border('+', '+', '+'))
        for var, desc, val, unit in fields:
            row = [
                f"{WHITE}{var:<{widths[0]}}{RESET}",
                f"{GREEN}{desc:<{widths[1]}}{RESET}",
                f"{ORANGE}{val:>{widths[2]}.4f}{RESET}",
                f"{ORANGE}{unit:<{widths[3]}}{RESET}",
            ]
            print('| ' + ' | '.join(row) + ' |')
        print(make_border('+', '+', '+') + "\n")

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

        rhol = (yl_vent / self.vl_vent) + ((1 - yl_vent) / self.vg_vent)  
        yl_0 = (rhol - (1/self.vg_0)) / (1/self.vl_0 - 1/self.vg_0)  
        return yl_0

    # Calculated the required tank volume based on the mass of hydrogen, initial liquid mass fraction, and density.
    def calculateTankVolume(self, rho_H2, m_H2, yl_0):
        """Calculate the required tank volume V from the input parameters."""

        V_tank = m_H2 / (rho_H2 * yl_0)  
        return V_tank 
    
    # Calculate the tank geometry (e.g., radius, length) based on the volume and design constraints.
    def calculateTankGeometry(self, V_tank, phi, psi, Lambda):
        """Calculate the tank geometry based on the volume and shape parameters."""
        c = (V_tank / (np.pi * phi * psi * ((2 * Lambda / (1 - Lambda)) + 4/3))) ** (1/3)
        a = phi * c
        b = psi * c
        ls = 2 * b * Lambda / (1 - Lambda)
        lt = ls + 2 * b

        # Cylindrical Surface Area
        if phi == 1.0 and psi == 1.0:
            A_body = 2 * np.pi * c * ls 
        else:
            h = ((a - c) / (a + c))**2
            p_ellipse = np.pi * (a + c) * (1 + 3*h / (10 + np.sqrt(4 - 3*h)))
            A_body = p_ellipse * ls

        # End-cap surface area
        if Lambda == 0:
            A_heads = 4 * np.pi * (c**2)
        else:
            p = 1.6075
            A_heads = 4*np.pi * (((a*c)**p + (a*b)**p + (c*b)**p) / 3)**(1/p)

        A_tank = A_body + A_heads

        return Geometry(a=a, b=b, c=c, ls=ls, lt=lt, V_tank=V_tank, A_tank=A_tank, ull=ullage_factor)
   


if __name__ == "__main__":
    # 1. Define inputs and call geom_design
    m_H2 = 600 # kg, mass of hydrogen to be stored
    p_fill = 1. # bar, fill pressure
    p_vent = 1.5 * p_fill # bar, venting pressure
    geom_design = GeomDesign(p_vent=p_vent, p_fill=p_fill, y_max=0.97)

    # 2. Initial liquid fraction at fill so that y_l = y_max at venting
    yl_0 = geom_design.calculateInitialLiquidMassFraction(yl_vent=geom_design.y_max)
    ullage_factor = 1 - yl_0

    # 3. Mean mixture density at venting (from CoolProp / fluids module)
    rho_H2_fill = PropsSI('D', 'P', p_fill*geom_design.BAR, 'T', 15, 'Hydrogen')
    rho_H2_vent = PropsSI('D', 'P', p_vent*geom_design.BAR, 'T', 15, 'Hydrogen')

    # 4. Calculate the required volume
    V_tank = geom_design.calculateTankVolume(rho_H2=rho_H2_vent, m_H2=m_H2, yl_0=yl_0)

    # 5. Dimensions for the chosen shape (area comes for free via the property)
    geom = geom_design.calculateTankGeometry(V_tank, phi=1.0, psi=1.0, Lambda=0.55)

    # 6. Print geometry summary
    geom.print_summary()
    