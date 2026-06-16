"""
Sizes the outer CFRP wall of the LH2 tank.
"""

import numpy as np
from dataclasses import dataclass

g = 9.81   # [m/s^2]


@dataclass(frozen=True)
class outerWall:
    Ex:       float   # longitudinal Young's modulus [GPa]
    Ey:       float   # transverse Young's modulus [GPa]
    nuxy:     float   # in-plane Poisson's ratio [-]
    Gxy:      float   # in-plane shear modulus [GPa]
    mCFRP:    float   # CFRP wall mass [kg]
    Rout:     float   # outer radius [m]
    tout:     float   # total wall thickness [m]
    sigma_x:  float   # applied axial compressive stress (9g load case) [GPa]
    sigma_cr: float   # imperfection-corrected critical buckling stress [GPa]
    Gamma:    float   # imperfection factor [-]

    def print_summary(self):
        fields = [
            ("Rout",     "Outer radius",          self.Rout,     "m"  ),
            ("tout",     "Wall thickness",         self.tout,     "m"  ),
            ("Ex",       "Long. modulus",          self.Ex,       "GPa"),
            ("Ey",       "Trans. modulus",         self.Ey,       "GPa"),
            ("nuxy",     "Poisson's ratio",        self.nuxy,     "-"  ),
            ("Gxy",      "Shear modulus",          self.Gxy,      "GPa"),
            ("mCFRP",    "CFRP mass",              self.mCFRP,    "kg" ),
            ("sigma_x",  "Applied Axial Stress",   self.sigma_x,  "GPa"),
            ("sigma_cr", "Critical Stress", self.sigma_cr, "GPa"),
            ("Gamma",    "Knockdown Factor",    self.Gamma,    "-"  ),
        ]

        RESET  = '\033[0m'
        CYAN   = '\033[96m'
        WHITE  = '\033[97m'
        GREEN  = '\033[92m'
        ORANGE = '\033[38;5;208m'
        BOLD   = '\033[1m'

        widths = [7, 24, 10, 5]

        def make_border(left, mid, right):
            return left + mid.join('-' * (w + 2) for w in widths) + right

        title_width = sum(widths) + 3 * len(widths) + (len(widths) - 1)
        print(f"\n{BOLD}{'OUTER WALL SUMMARY':^{title_width}}{RESET}")
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


class outerSizing:
    def __init__(self, E, nu, G, rhoCFRP, angles, f_fibre=0.65, tply=0.000215):
        self.E       = E
        self.nu      = nu
        self.G       = G
        self.angles  = angles
        self.tply    = tply
        self.rhoBulk = f_fibre * rhoCFRP['fibre'] + (1 - f_fibre) * rhoCFRP['resin']

        E1   = E['E1'];  E2 = E['E2']
        G12  = G['G12']
        nu12 = nu['nu12']
        nu21 = nu12 * (E2 / E1)
        den  = 1 - nu12 * nu21

        self._Qred = np.array([
            [E1 / den,          nu12 * E2 / den, 0  ],
            [nu12 * E2 / den,   E2 / den,        0  ],
            [0,                 0,               G12],
        ])

    def _Qbar(self, theta):
        r = np.radians(theta)
        c, s = np.cos(r), np.sin(r)
        c2, s2, cs = c**2, s**2, c * s
        T1 = np.array([[c2, s2, 2*cs], [s2, c2, -2*cs], [-cs, cs, c2-s2]])
        T2 = np.array([[c2, s2, cs],   [s2, c2, -cs],   [-2*cs, 2*cs, c2-s2]])
        return np.linalg.inv(T1) @ self._Qred @ T2

    def _ABD(self):
        n    = len(self.angles)
        tout = n * self.tply
        A    = np.zeros((3, 3))
        for angle in self.angles:
            A += self._Qbar(angle) * self.tply

        z = np.linspace(-tout / 2, tout / 2, n + 1)
        D = np.zeros((3, 3))
        for k, angle in enumerate(self.angles):
            D += (1 / 3) * self._Qbar(angle) * (z[k + 1]**3 - z[k]**3)

        return A, D, tout

    def size(self, Rin, tMLI, twall, ls, m_empty):
        A, _, tout = self._ABD()
        a    = np.linalg.inv(A)

        Ex   = 1 / (tout * a[0, 0])   # [GPa]
        Ey   = 1 / (tout * a[1, 1])   # [GPa]
        nuxy = -a[0, 1] / a[0, 0]
        Gxy  = 1 / (tout * a[2, 2])   # [GPa]

        Rout  = Rin + twall + tMLI + tout
        Aout  = 4 * np.pi * Rout**2 + 2 * np.pi * Rout * ls
        mCFRP = tout * self.rhoBulk * Aout

        # Cylinder buckling check (NASA imperfection approach, 9g load case)
        delta    = Rout / 1200
        Gamma    = np.sqrt(1 - nuxy**2) * (delta / tout)
        sigma_x  = ((9 * g * m_empty) / (2 * np.pi * Rout * tout)) / 1e9
        sigma_cr = Ex / np.sqrt(3 * (1 - nuxy**2)) * (tout / Rout) * Gamma

        return outerWall(Ex=Ex, Ey=Ey, nuxy=nuxy, Gxy=Gxy,
                         mCFRP=mCFRP, Rout=Rout, tout=tout,
                         sigma_x=sigma_x, sigma_cr=sigma_cr, Gamma=Gamma)


if __name__ == "__main__":
    Al2219T87 = {'E': 85.46e9, 'nu': 0.3184, 'S': 526e6, 'S_t': 717e6, 'density': 2825}

    E_ITS50      = {'E1': 168.0, 'E2': 9.0,  'E3': 9.0}
    nu_ITS50     = {'nu12': 0.1, 'nu13': 0.1, 'nu23': 0.3}
    G_ITS50      = {'G12': 5.0,  'G13': 5.0,  'G23': 3.7}
    rhoCFRP_ITS50 = {'fibre': 1800, 'resin': 1195}

    angles = [0, 45, -45, 90, 90, -45, 45, 0]

    Rin   = 0.8913   # [m]
    twall = 0.002    # [m]
    tMLI  = 0.008218 # [m]
    ls    = 2.1789   # [m]
    

    sizing = outerSizing(E_ITS50, nu_ITS50, G_ITS50, rhoCFRP_ITS50, angles)
    result = sizing.size(Rin=Rin, tMLI=tMLI, twall=twall, ls=ls)
    result.print_summary()
