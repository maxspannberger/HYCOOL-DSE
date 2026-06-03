"""
Main storage class for sizing the liquid hydrogen tank.
"""

# Import statements
from storageParameters import *
from CoolProp import AbstractState
from CoolProp.CoolProp import PhaseSI, PropsSI, get_global_param_string
import CoolProp.CoolProp as CoolProp
from CoolProp.HumidAirProp import HAPropsSI
import numpy as np


# Generate the object oriented class
class MainStorage:
    def __init__(self, tank_volume, tank_inner_diameter, tank_outer_diameter, tank_length, tank_insulation_thickness, tank_wall_thickness):
        self.pVent = np.linspace(0, pLH2, pLH2 + 10) # bar, venting pressure range for the tank
    
    def calculate_required_LH2_volume(self, mLH2, TLH2_tank, pLH2_tank):
        """Calculate the required volume of liquid hydrogen based on the density of liquid hydrogen at a given temperature and pressure"""
        density_LH2 = PropsSI("D", "T", TLH2_tank, "P", pLH2_tank, "hydrogen")
        VLH2 = mLH2 / density_LH2
        return VLH2

    def calculate_gravimetric_index(self, mLH2, mWall, mIns):
        """Calculate the gravimetric index of the tank based on the mass of liquid hydrogen and the mass of the tank"""
        gravimetric_index = mLH2 / (mLH2 + mWall + mIns)
        return gravimetric_index


if __name__ == "__main__":    # Example usage
    tank_volume = 1000 # m^3
    tank_inner_diameter = 5 # m
    tank_outer_diameter = 6 # m
    tank_length = 10 # m
    tank_insulation_thickness = 0.1 # m
    tank_wall_thickness = 0.05 # m

    main_storage = MainStorage(tank_volume, tank_inner_diameter, tank_outer_diameter, tank_length, tank_insulation_thickness, tank_wall_thickness)
    

    