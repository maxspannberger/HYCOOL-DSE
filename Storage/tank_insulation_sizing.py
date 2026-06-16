"""
Size the MLI insulation layer of the LH2 tank.
"""
import numpy as np
from CoolProp.CoolProp import PropsSI
from dataclasses import dataclass

FLUID = "parahydrogen"


@dataclass(frozen=True)
class insulation:
    hMLI:  float   # effective heat transfer coefficient [W/(m^2 K)]
    tMLI:  float   # MLI thickness [m]
    N:     float   # number of MLI layers [-]
    Nbar:  float   # optimum layer density [layers/cm]
    qtank: float   # raw MLI heat flux (N=1 basis) [W/m^2]
    qMLI: float   # allowable heat flux [W/m^2]
    mMLI:  float   # MLI mass [kg]

    def print_summary(self):
        fields = [
            ("Nbar",  "Optimum layer density",   self.Nbar,  "layers/cm"),
            ("N",     "Number of MLI layers",    self.N,     "-"        ),
            ("tMLI",  "MLI thickness",           self.tMLI,  "m"        ),
            ("qtank", "Allowable MLI heat flux",       self.qtank, "W/m^2"    ),
            ("qMLI", "Actual MLI heat flux",     self.qMLI, "W/m^2"    ),
            ("hMLI",  "Heat transfer coeff.",    self.hMLI,  "W/(m^2 K)"),
            ("mMLI",  "MLI mass",               self.mMLI,  "kg"       ),
        ]

        RESET  = '\033[0m'
        CYAN   = '\033[96m'
        WHITE  = '\033[97m'
        GREEN  = '\033[92m'
        ORANGE = '\033[38;5;208m'
        BOLD   = '\033[1m'

        widths = [7, 24, 10, 10]

        def make_border(left, mid, right):
            return left + mid.join('-' * (w + 2) for w in widths) + right

        title_width = sum(widths) + 3 * len(widths) + (len(widths) - 1)
        print(f"\n{BOLD}{'MLI INSULATION SUMMARY':^{title_width}}{RESET}")
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


class insulationSizing:
    def __init__(self, Ts=298, Tc=20.3, P=1e-4, rhoMLI=20):
        self.Ts     = Ts       # hot wall temperature [K]
        self.Tc     = Tc       # cold wall temperature [K]
        self.P      = P        # interstitial pressure [torr]
        self.rhoMLI = rhoMLI   # MLI bulk density [kg/m^3]

        # Lockheed MLI empirical constants
        self.Cs         = 7.30e-8
        self.Cg         = 1.46e4
        self.Cr         = 7.07e-10
        self.emissivity = 0.043

        self.Nbar = self._calcNbar()

    def _calcNbar(self):
        Ts, Tc = self.Ts, self.Tc
        num = (self.Cr * self.emissivity * (Ts**4.67 - Tc**4.67)
               + self.Cg * self.P * (Ts**0.52 - Tc**0.52))
        den = 0.78 * self.Cs * (Ts**2 - Tc**2)
        return (num / den) ** (1 / 2.63)

    def _calcQleak(self):
        Ts, Tc = self.Ts, self.Tc
        qs = (self.Cs * self.Nbar**2.63 * (Ts**2 - Tc**2)) / 2
        qr = (self.Cr * self.emissivity * (Ts**4.67 - Tc**4.67))
        qg = self.Cg * self.P * (Ts**0.52 - Tc**0.52)
        return qs + qr + qg

    def size(self, Qleak, Atank, Rtank, ls):
        qleak = self._calcQleak()
        qtank = Qleak / Atank
        N     = np.ceil(qleak / qtank)
        qMLI  = qleak / N
        tMLI  = (N / self.Nbar) * 0.01   # Nbar in layers/cm -> m
        Aout  = 4 * np.pi * (Rtank + tMLI)**2 + 2 * np.pi * (Rtank + tMLI) * ls
        hMLI  = qMLI / (self.Ts - self.Tc)
        mMLI  = self.rhoMLI * tMLI * Aout

        return insulation(hMLI=hMLI, tMLI=tMLI, N=N, Nbar=self.Nbar,
                          qtank=qtank, qMLI=qMLI, mMLI=mMLI)


if __name__ == "__main__":
    BAR   = 1e5
    mLH2  = 600
    Pfill = 1.0 * BAR
    Pvent = 1.75 * Pfill
    rhoLH2 = 70
    T=15
    Ef = PropsSI('U', 'P', Pfill, 'T', 15, FLUID)
    Ei = PropsSI('U', 'P', Pvent, 'T', 20.3, FLUID)
    Qleak = mLH2 * (Ei - Ef) / (24 * 3600)

    ls    = 2.1789
    Rtank = 0.8913
    Atank = 4 * np.pi * Rtank**2 + 2 * np.pi * Rtank * ls

    ins  = insulationSizing()
    insulationData = ins.size(Qleak=Qleak, Atank=Atank, Rtank=Rtank, ls=ls)
    insulationData.print_summary()
