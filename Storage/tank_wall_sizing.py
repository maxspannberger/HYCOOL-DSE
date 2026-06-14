"""
Sizes the inner aluminium lining of the tank and the outer CFRP wall. 
"""

from CoolProp.CoolProp import PropsSI
import numpy as np
from dataclasses import dataclass
import properties as props

FLUID = "parahydrogen" 

Ufill = PropsSI('U', 'P', 1.75e5, 'T', 15, FLUID)
Uvent = PropsSI('U', 'P', 1.75e5 * 1.5, 'T', 15, FLUID)

print(Ufill)
print(Uvent)

print((Ufill - Uvent)/24)