from CoolProp.CoolProp import PropsSI
from scipy.integrate import quad
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional
from geomDesign import GeomDesign, Geometry
import properties as props
import numpy as np


FLUID = "parahydrogen"
BAR = 1e5  # Pa per bar


class insulationDesign():
    def __init__(self):
        # Lockheed Parameters
        self.Cs = 1.93e-6 
        self.Cg = 1
        self.Cr = 3.88e-10
        self.epsilon = 0.03
        self.p = 1.2e-3
        pass

    def MaximumHeatLeakRate(self, P_fill, P_vent, T_fill, T_vent, m_LH2, tau_hold):
        E_i = PropsSI('U', 'P', P_fill, 'T', T_fill, 'parahydrogen') * m_LH2 # Internal energy at filling
        E_f = PropsSI('U', 'P', P_vent, 'T', T_fill, 'parahydrogen') * m_LH2 # Internal energy at venting pressure 

        Q_leak = (E_i - E_f) / tau_hold
        return Q_leak

    def TankSurfaceTemperature(self, Q_Leak, h_air, T_0, A_out):
        T_S = T_0 - (Q_Leak)/(h_air * A_out) 

        return T_S
    
    def AirThermalConductivity(self, geom: Geometry, T_s, T_0, P_0):
        D = geom.b * 2  # [m]
        g = 9.81  # [m/s^2]
        T_film = (T_s + T_0) / 2 

        # Thermodynamic Properties
        Pr = PropsSI('PRANDTL', 'P', P_0, 'T', T_film, FLUID)
        beta = PropsSI('isobaric_expansion_coefficient', 'P', P_0, 'T', T_film, FLUID)
        mu = PropsSI('V', 'T', T_film, 'P', P_0, FLUID)
        rho = PropsSI('D', 'T', T_film, 'P', P_0, FLUID)
        kinematic_viscosity = mu / rho

        Ral = (g * beta * abs(T_s - T_0) * (D ** 3) * Pr) / (kinematic_viscosity ** 2)
        Nu = (0.60 + ( (0.387 * (Ral ** (1/6))) / ((1 + (0.559 / Pr) ** (9/16)) ** (8/27)) ) ) ** 2
        k_air = PropsSI('conductivity', 'T', T_film, 'P', P_0, FLUID)

        h_air = (Nu * k_air) / D

        return h_air

    def IterateInsulationDesign(self, geom: Geometry, Q_Leak, T_0, P_0, T_wall, N_bar,
                                d_ins_0, T_s_0, tol_d=1e-6, tol_T=1e-6, max_iter=200):
        d_ins = d_ins_0
        T_s = T_s_0

        for _ in range(max_iter):
            A_out = self.outerSurfaceArea(geom, d_ins)
            h_air = self.AirThermalConductivity(geom, T_s, T_0, P_0)

            T_s_new = self.TankSurfaceTemperature(Q_Leak, h_air, T_0, A_out)
            d_ins_new = self.insulationThickness(N_bar, Q_Leak, A_out, T_s_new, T_wall)

            converged = abs(T_s_new - T_s) < tol_T and abs(d_ins_new - d_ins) < tol_d
            T_s, d_ins = T_s_new, d_ins_new

            if converged:
                break

        return d_ins, T_s

    def insulationThickness(self, N_bar, Q_leak, A, T_s, T_wall):
        Cs = self.Cs  
        Cg = self.Cg 
        Cr = self.Cr 
        epsilon = self.epsilon 
        p = self.p 
        Tm = (T_s + T_wall) / 2

        q_leak = Q_leak / A
        A1 = Cs * Tm * (N_bar ** 2.63) * (T_s - T_wall)
        A2 = Cr * epsilon * ((T_s ** 4.67) - (T_wall ** 4.67)) + Cg * p * ((T_s ** 0.52) - (T_wall ** 0.52))
        A3 = q_leak + A1 + A2

        N_up = (A3 + np.sqrt((A3 ** 2) - 4 * q_leak * A2)) / (2 * q_leak)
        N_low = (A3 - np.sqrt((A3 ** 2) - 4 * q_leak * A2)) / (2 * q_leak)

        print(f"N_up: {N_up}")
        print(f"N_low: {N_low}")
        N = min(N_up, N_low)

        d_ins = N / N_bar

        return d_ins

    def outerSurfaceArea(self, geom: Geometry, d_ins):
        d_wall = 0.01 # Wall thickness
        Ain = geom.A_tank
        ls = geom.ls
        D = geom.b * 2  # [m]

        A_out = Ain + np.pi * (d_ins * ls + d_ins ** 2 + 2 * D * d_ins + 2 * d_wall * d_ins)

        return A_out


if __name__ == "__main__":
    # Mission inputs
    tau_hold = 24 * 3600   # [s] max no-vent holding time (24 hours)
    m_LH2 = 250            # [kg] stored LH2 mass

    p_fill, p_vent = 1.0, 1.5      # [bar]
    T_fill, T_vent = 15.0, 20.0    # [K]

    gd = GeomDesign(p_vent=p_vent, p_fill=p_fill, y_max=0.97)
    yl_0 = gd.calculateInitialLiquidMassFraction(yl_vent=0.97)
    rho_lh2 = PropsSI('D', 'P', p_fill * BAR, 'T', T_fill, FLUID)
    V = gd.calculateTankVolume(rho_H2=rho_lh2, m_H2=m_LH2, yl_0=yl_0)
    geom = gd.calculateTankGeometry(V, phi=1.0, psi=1.0, Lambda=0.75)

    ins = insulationDesign()

    Q_leak = ins.MaximumHeatLeakRate(
        P_fill=p_fill * BAR, P_vent=p_vent * BAR,
        T_fill=T_fill, T_vent=T_vent,
        m_LH2=m_LH2, tau_hold=tau_hold,
    )

    d_ins, T_s = ins.IterateInsulationDesign(
        geom=geom, Q_Leak=Q_leak,
        T_0=293.0, P_0=101325.0,
        T_wall=T_vent, N_bar=30.0,
        d_ins_0=0.005, T_s_0=250.0,
    )

    print(f"Q_leak = {Q_leak:.6f} W")
    print(f"d_ins  = {d_ins:.6g} m")
    print(f"T_s    = {T_s:.6g} K")