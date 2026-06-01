# Import Statements
from storageParameters import *
from CoolProp import AbstractState
from CoolProp.CoolProp import PhaseSI, PropsSI, get_global_param_string
import CoolProp.CoolProp as CoolProp
from CoolProp.HumidAirProp import HAPropsSI
from scipy.optimize import brentq
import numpy as np

def calculateLiquidMassFraction(P, yl0):
    """Calculate the liquid mass fraction at a given pressure based on the initial liquid mass fraction and the pressure ratio"""
    rho0 = yl0 / vl0 + (1.0 - yl0) / vg0
    yl = (rho0 - (1/vg0)) / (1/vl0 - 1/vg0)
    return yl 

def calculateInitialLiquidMassFraction(Pvent, y_l_vent=0.97, P0=1.0 * BAR, bracket=(0.50, 0.999)):
    """Calculates the initial liquid mass fraction (yl0) to achieve a mass fraction of 0.97 at a given venting pressure Pvent"""
    if Pvent <= P0:
        raise ValueError(
            f"PPvent ({Pvent/BAR:.3f} bar) must exceed fill pressure "
            f"P0 ({P0/BAR:.3f} bar)."
        )

    # Residual = how far the forward-mapped liquid fraction at P_vent sits
    # from the target. We want this to be zero.
    def residual(y_l0):
        return calculateLiquidMassFraction(Pvent, y_l0) - y_l_vent

    lo, hi = bracket
    f_lo, f_hi = residual(lo), residual(hi)

    # brentq needs the root bracketed: the residual must change sign across
    # [lo, hi]. If it doesn't, the target isn't reachable in that range.
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Target y_l={y_l_vent} not bracketed by y_l0 in [{lo}, {hi}] "
            f"at P_vent={Pvent/BAR:.3f} bar "
            f"(residuals {f_lo:.3g}, {f_hi:.3g}). Widen the bracket."
        )

    yl0 = brentq(residual, lo, hi, xtol=1e-10, rtol=1e-12)
    return yl0

def calculateTankVolume(mLH2, yl0, VLH2):
    """Calculate the required tank volume based on the volume of liquid hydrogen and the initial liquid mass fraction"""
    
    Vtank = VLH2 / yl0
    return Vtank