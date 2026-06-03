"""
Sizes the tank and calculates the dimensions of the tank. 
"""

from CoolProp.CoolProp import PropsSI
from scipy.integrate import quad
from scipy.optimize import brentq
import numpy as np
from dataclasses import dataclass
import properties as props

class ThermalDesign:
    def __init__(self):
        pass

    def internalConvection(self):
        pass

    def thermalConduction(self):
        pass

    def externalConvection(self):
        pass