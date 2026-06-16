"""
Sizes the anti-slosh ring baffles inside the LH2 tank cylindrical section.
"""

import numpy as np
from dataclasses import dataclass

g = 9.81   # [m/s^2]
C = 1.4424  # annular plate constant: free inner edge, fixed outer edge, Rin/Rout = 0.5


@dataclass(frozen=True)
class baffles:
    n:       int    # number of baffles [-]
    t:       float  # baffle thickness [m]
    Rin_b:   float  # baffle inner radius [m]
    Rout_b:  float  # baffle outer radius [m]
    mBaffle: float  # total baffle mass [kg]

    def print_summary(self):
        fields = [
            ("n",       "Number of baffles",    self.n,       "-" ),
            ("t",       "Baffle thickness",      self.t,       "m" ),
            ("Rin_b",   "Inner radius",          self.Rin_b,   "m" ),
            ("Rout_b",  "Outer radius",          self.Rout_b,  "m" ),
            ("mBaffle", "Total baffle mass",     self.mBaffle, "kg"),
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
        print(f"\n{BOLD}{'BAFFLE SIZING SUMMARY':^{title_width}}{RESET}")
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


class baffleSizing:
    def __init__(self, mat, mLH2):
        self.mat  = mat
        self.mLH2 = mLH2

    def size(self, R, ls, s, rho_LH2):
        S   = self.mat['S']
        rho = self.mat['density']

        # Ring baffle geometry: Rin/Rout = 0.5
        Rout_b = R
        Rin_b  = 0.5 * R
        A_b    = np.pi * (Rout_b**2 - Rin_b**2)

        # Number of baffles — rounded up to nearest odd so centre baffle is a full disk separator
        n = int(np.ceil(ls / s))
        if n % 2 == 0:
            n += 1

        # Reduce n (keeping odd) until total spread n*s fits within ls
        while n * s >= ls and n > 1:
            n -= 2

        # Baffle thickness: t = R * sqrt(C * rho_LH2 * 9g / (S * n))
        t = R * np.sqrt(C * rho_LH2 * 9 * g / (S * n))
        t = max(t, 0.002)  # 2 mm minimum

        # Centre baffle: full disk (no hole); remaining n-1 are annular rings
        A_centre  = np.pi * Rout_b**2
        mBaffle = rho * t * (A_centre + (n - 1) * A_b)

        return baffles(n=n, t=t, Rin_b=Rin_b, Rout_b=Rout_b, mBaffle=mBaffle)


if __name__ == "__main__":
    Al2219T87 = {'E': 85.46e9, 'nu': 0.3184, 'S': 526e6, 'S_t': 717e6, 'density': 2825}
    mLH2      = 600   # [kg]
    R         = 0.8725  # [m]
    ls        = 2.6176  # [m]

    rho_LH2 = 70.8  # [kg/m^3] saturated LH2 density at fill pressure
    s       = 0.5   # [m] baffle spacing
    sizing     = baffleSizing(Al2219T87, mLH2)
    baffleData = sizing.size(R=R, ls=ls, s=s, rho_LH2=rho_LH2)
    baffleData.print_summary()
