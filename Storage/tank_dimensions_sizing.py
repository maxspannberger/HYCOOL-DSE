"""
Sizes the tank and calculates the dimensions of the tank. 
"""

from CoolProp.CoolProp import PropsSI
import numpy as np
from dataclasses import dataclass, field, replace
import properties as props

FLUID = "parahydrogen"   

@dataclass(frozen=True)
class dimensions:
    R: float        # semi-axis (φ·c)
    Rin: float
    ls: float      # straight shell length
    lt: float      # total length
    V_tank: float      # tank volume
    A_in: float
    ull: float
    t_in: float = field(default=None)
    m_in: float = field(default=None)

    def print_summary(self):
        fields = [
            ("R", "LH2 Volume Radius", self.R, "m" ),
            ("Rin",      "Tank Radius with liner",        self.Rin,       "m"   ),
            ("ls",     "Straight shell length",    self.ls,      "m"   ),
            ("lt",     "Total length",             self.lt,      "m"   ),
            ("V_tank", "Tank volume",              self.V_tank,  "m^3" ),
            ("A_in", "Total liner surface area",       self.A_in,  "m^2" ),
            ("U", "Ullage Factor", self.ull, "-"),
            ("t_in", "Inner liner thickness", self.t_in, "m"),
            ("m_in", "Inner liner mass",      self.m_in, "kg"),
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
        print(f"\n{BOLD}{'TANK INNER SIZING SUMMARY':^{title_width}}{RESET}")
        print(make_border('+', '+', '+'))
        print('|' + '|'.join(f" {CYAN}{h:^{w}}{RESET} " for h, w in zip(['Var', 'Description', 'Value', 'Unit'], widths)) + '|')
        print(make_border('+', '+', '+'))
        for var, desc, val, unit in fields:
            row = [
                f"{WHITE}{var:<{widths[0]}}{RESET}",
                f"{GREEN}{desc:<{widths[1]}}{RESET}",
                f"{ORANGE}{f'{val:.4f}' if val is not None else 'N/A':>{widths[2]}}{RESET}",
                f"{ORANGE}{unit:<{widths[3]}}{RESET}",
            ]
            print('| ' + ' | '.join(row) + ' |')
        print(make_border('+', '+', '+') + "\n")

class dimensionSizing:
    def __init__(self, p_fill, p_vent, y_max=0.97):

        # UNIT TEST: Check that p_vent > p_fill
        if not p_vent > p_fill:
            raise ValueError(f"p_vent ({p_vent}) must be greater than p_fill ({p_fill})")

        self.p_vent = p_vent
        self.p_fill = p_fill
        self.y_max = y_max
        self.vl_0 = props.saturated_specific_volumes(self.p_fill, FLUID)[0]
        self.vg_0 = props.saturated_specific_volumes(self.p_fill, FLUID)[1]
        self.vl_vent = props.saturated_specific_volumes(self.p_vent, FLUID)[0] 
        self.vg_vent = props.saturated_specific_volumes(self.p_vent, FLUID)[1]
        self.rhol = (self.y_max / self.vl_vent) + ((1 - self.y_max) / self.vg_vent) 

    # Calculate the liquid mass fraction at a given pressure and initial liquid mass fraction.
    def calculateInitialLiquidMassFraction(self):        
        """Calculate the liquid mass fraction at a given pressure based on the initial liquid mass fraction and the pressure ratio."""
        yl_0 = (self.rhol - (1/self.vg_0)) / (1/self.vl_0 - 1/self.vg_0) 
        ullage_factor = 1.0 - yl_0
        print(ullage_factor)
        return yl_0, ullage_factor

    # Calculated the required tank volume based on the mass of hydrogen, initial liquid mass fraction, and density.
    def calculateTankVolume(self, rho_H2, m_H2, yl_0):
        """Calculate the required tank volume V from the input parameters."""

        V_tank = m_H2 / (rho_H2 * yl_0)
        return V_tank
    
    # Size the inner liner
    def sizeInnerLiner(self, mat, pint, pext, R):
        
        S = mat['S']
        D = 2 * R
        t_in = (D * (pint - pext)) / (2 * S)
        
        if t_in < 0.002:
            print(f"The thickness of the inner liner, {t_in * 1e3} [mm], is too thin. It will be rounded up to 2.0 [mm].")
            t_in = 0.002

        Rin = R + t_in

        return t_in, Rin
    
    # Calculate the tank geometry (e.g., radius, length) based on the volume and design constraints.
    def calculateTankGeometry(self, mat, pint, pext, V_tank, ullage_factor, Lambda, phi=1.0, psi=1.0):
        """Calculate the tank geometry based on the volume and shape parameters."""
        c = (V_tank / (np.pi * phi * psi * ((2 * Lambda / (1 - Lambda)) + 4/3))) ** (1/3)
        a = phi * c
        b = psi * c
        ls = 2 * b * Lambda / (1 - Lambda)
        lt = ls + 2 * b
        R = c
        t_in, Rin = self.sizeInnerLiner(mat, pint, pext, R)
        

        # Cylindrical Surface Area
        if phi == 1.0 and psi == 1.0:
            A_body = 2 * np.pi * Rin * ls 
        else:
            h = ((a - c) / (a + c))**2
            p_ellipse = np.pi * (a + c) * (1 + 3*h / (10 + np.sqrt(4 - 3*h)))
            A_body = p_ellipse * ls

        # End-cap surface area
        if Lambda == 0:
            A_heads = 4 * np.pi * (Rin**2)
        else:
            p = 1.6075
            A_heads = 4*np.pi * (((a*c)**p + (a*b)**p + (c*b)**p) / 3)**(1/p)

        A_in = A_body + A_heads
        m_in = t_in * A_in * mat['density']


        return dimensions(R=c, Rin=Rin, ls=ls, lt=lt, V_tank=V_tank, A_in = A_in, ull=ullage_factor, t_in=t_in, m_in=m_in)
    
    
    
   


if __name__ == "__main__":
    # 1. Define inputs and call d
    BAR = 1e5  # bar to pascal conversion
    m_H2 = 600 # kg, mass of hydrogen to be stored
    p_fill = 1.0 * BAR # bar, fill pressure
    p_vent = 1.75 * BAR
    pext_SLS = 101325  # [Pa] Ambient pressure at SLS conditions
    pext_FL250 = 37600  # [Pa] Ambient pressure at FL250
    Al2219T87 = {'E': 85.46e9, 'nu': 0.3184, 'S': 526e6, 'S_t': 717e6, 'density': 2825}
    d = dimensionSizing(p_vent=p_vent, p_fill=p_fill)

    # 2. Initial liquid fraction at fill so that y_l = y_max at venting
    yl_0, ull = d.calculateInitialLiquidMassFraction()


    # 3. Mean mixture density at venting (from CoolProp / fluids module)
    rho_H2_fill = PropsSI('D', 'P', p_fill, 'T', 15, 'Parahydrogen')
    rho_H2_vent = PropsSI('D', 'P', p_vent, 'T', 15, 'Parahydrogen')

    # 4. Calculate the required volume
    V_tank = d.calculateTankVolume(rho_H2=rho_H2_vent, m_H2=m_H2, yl_0=yl_0)

    # 5. Dimensions for the chosen shape (area comes for free via the property)
    geom = d.calculateTankGeometry(V_tank, ull, Lambda=0.60)
    geom = d.sizeInnerLiner(Al2219T87, p_fill, pext_FL250, V_tank, ull, Lambda=0.60)
    # 6. Print geometry summary
    geom.print_summary()
    