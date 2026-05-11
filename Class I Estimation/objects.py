import numpy as np
import scipy as sp
import math
import os
import matplotlib.pyplot as plt
from parameters import lift_coefficients
from parameters import propulsion_parameters
from parameters import aerodynamic_parameters
from parameters import flight_parameters

class Atmosphere:
    G = 9.80665          
    R = 287.0528         
    GAMMA = 1.4          
    S = 110.4            
    MU_REF = 1.716e-5    
    T_REF = 273.15       

    def __init__(self, altitude_m, isa_offset_c=0.0):
        if altitude_m < 0:
            raise ValueError("Altitude must be zero or positive.")
        self.altitude = altitude_m
        self.isa_offset = isa_offset_c
        self._calculate_state()

    def _calculate_state(self):
        if self.altitude <= 11000.0:
            Tb = 288.15
            Pb = 101325.0
            a = -0.0065
            self.temperature = Tb + a * self.altitude
            self.pressure = Pb * (self.temperature / Tb) ** (-self.G / (a * self.R))
        elif self.altitude <= 20000.0:
            Tb = 216.65
            Pb = 22632.1
            hb = 11000.0
            self.temperature = Tb
            self.pressure = Pb * math.exp(-self.G * (self.altitude - hb) / (self.R * Tb))
        else:
            raise NotImplementedError("Model only supports altitudes up to 20,000 meters.")
            
        self.temperature += self.isa_offset
        self.density = self.pressure / (self.R * self.temperature)
        self.speed_of_sound = math.sqrt(self.GAMMA * self.R * self.temperature)
        self.dynamic_viscosity = (self.MU_REF * ((self.temperature / self.T_REF) ** 1.5) * ((self.T_REF + self.S) / (self.temperature + self.S)))
        self.kinematic_viscosity = self.dynamic_viscosity / self.density

    def set_altitude(self, new_altitude_m):
        self.altitude = new_altitude_m
        self._calculate_state()
    
class MatchingDiagram:
    def __init__(self, Vs0=55.0, Vapp=57, LFL=1500, c=0, G=0.0024, TO_field_length=1000):
        self.lift_coefficients = lift_coefficients
        self.propulsion_parameters = propulsion_parameters
        self.aerodynamic_parameters = aerodynamic_parameters
        self.flight_parameters = flight_parameters

        self.Ne = self.propulsion_parameters["Ne"]  
        self.A = self.aerodynamic_parameters["A"]  
        self.e = self.aerodynamic_parameters["e"]  
        self.CD0 = self.aerodynamic_parameters["CD0"]  
        self.kP = self.propulsion_parameters["kP"]
        self.MTOW = self.flight_parameters["MTOW"]  
        self.MCR = self.flight_parameters['MCR']  
        self.hcr = self.flight_parameters['Cruise_altitude'] 
        self.eta_prop = self.propulsion_parameters["eta_prop"]  

        self.gamma = Atmosphere.GAMMA
        self.R = Atmosphere.R
        self.g = Atmosphere.G
        self.S = Atmosphere.S
        self.MU_REF = Atmosphere.MU_REF
        self.T_REF = Atmosphere.T_REF

        self.beta = 0.7 
        self.CLFL = 0.45 
        self.h2 = 18 
        self.kT = 0.85 

        self.h_to = 10.7  
        self.mu_prime = .010 * lift_coefficients['CL_max_TO'] + .02
        self.CL_2 = .694 * lift_coefficients['CL_max_TO']
        self.BFL = 1800 
        self.Delta_S_TO = 200 

        self.density_cruise = Atmosphere(self.flight_parameters['Cruise_altitude']).density
        self.pressure_cruise = Atmosphere(self.flight_parameters['Cruise_altitude']).pressure
        self.temperature_cruise = Atmosphere(self.flight_parameters['Cruise_altitude']).temperature

        self.Vs0 = 55.0 
        self.Vapp = Vapp 
        self.LFL = LFL 
        
        self.V_cr = self.MCR * Atmosphere(self.flight_parameters['Cruise_altitude']).speed_of_sound 
        self.c = c 
        self.G = G 
        self.LTO = self.flight_parameters['TO_field_length'] 
        self.sigma = self.density_cruise / Atmosphere(0).density 
        self.Dp = 4.0 

        self.W_land = 25000  
        self.S_land = 2200 
        self.W_TO = 32000 
        self.f_land = 1.67  
        self.h_land = 15.3  
        self.a_g = 0.4  

        self.g_climb = 0.024  
        self.W_S = np.linspace(1, 5000, 1000) 

    def calculate_matching(self, atm: Atmosphere):
        P_W_cruise = self.eta_prop * (1 / self.beta) * np.float_power(((self.CD0 * 0.5 * self.density_cruise * np.power(self.V_cr, 3))/(self.beta * self.W_S)) + ((self.beta * (self.W_S)) / (np.pi * self.A * self.e * 0.5 * self.density_cruise * self.V_cr)), -1)
        W_P_cruise = 1 / P_W_cruise

        P_W_TO = (1 / np.float_power(self.kP, 1.5)) * np.sqrt(self.MTOW / (self.sigma * self.Ne * np.square(self.Dp))) * (self.mu_prime + 1 / ((1.159*(self.BFL - (self.Delta_S_TO/np.sqrt(self.sigma))))/(self.W_S/(atm.density*self.g*self.CL_2)+ self.h_to) - 2.7))
        W_P_TO = 1 / P_W_TO 

        W_S_land = (1 / (self.W_land/self.W_TO)) * ((self.S_land/(self.f_land * self.h_land)) - 10) * ((self.h_land * atm.density * self.g * self.lift_coefficients['CL_max_L'])/(1.52/self.a_g + 1.69))
        W_S_min = 0.5 * atm.density * self.lift_coefficients['CL_max_L'] * np.square(self.Vs0)
        
        CD = self.CD0 + self.CL_2 / (np.pi * self.A * self.e) 
        CL = self.CL_2
        W_P_climb = ((self.Ne - 1)/self.Ne) * self.eta_prop * (1 / self.beta) * (1 / (self.g_climb + (CD / CL))) * np.sqrt((self.density_cruise / 2)*(CL / (self.beta * self.W_S)))  
    
        W_P_Curves = {
                "Cruise Speed": W_P_cruise,
                "Take-off Distance": W_P_TO,
                "Climb Gradient": W_P_climb
        }
        W_S_Curves = {
                "Landing Distance": W_S_land,
                "Minimum Speed": W_S_min
        }
        return W_P_Curves, W_S_Curves
        
    def get_design_point(self, W_P_Curves, W_S_Curves):
        """
        Mathematically finds the optimum W/S and W/P.
        """
        # --- ERROR CATCHER ---
        # If the arguments were accidentally swapped in mainWeight.py, W_S_Curves 
        # will contain arrays. This swaps them back automatically to prevent crashes!
        if any(isinstance(v, np.ndarray) and v.size > 1 for v in W_S_Curves.values()):
            W_P_Curves, W_S_Curves = W_S_Curves, W_P_Curves
        
        # 1. The maximum allowable Wing Loading is the strictest (smallest) vertical limit.
        # We wrap them in float() to guarantee no numpy-array comparison errors.
        valid_W_S_limits = [float(v) for v in W_S_Curves.values()]
        optimum_W_S = min(valid_W_S_limits)
        
        # 2. Find the index in our W_S array closest to this optimum_W_S
        idx = (np.abs(self.W_S - optimum_W_S)).argmin()
        
        # 3. The maximum allowable Power Loading is the strictest (lowest) performance curve
        optimum_W_P = min(float(curve[idx]) for curve in W_P_Curves.values())
        
        return optimum_W_S, optimum_W_P
    
    def generate_diagram(self, W_P_Curves, W_S_Curves, design_point=None):
        plt.figure(figsize=(10, 6))
        for curve_name, curve_data in W_P_Curves.items():
            plt.plot(self.W_S, curve_data, label=curve_name)
    
        for curve_name, curve_data in W_S_Curves.items():
            plt.vlines(curve_data, ymin=0, ymax=1.0, label=curve_name, linestyles='dashed')
            
        if design_point:
            opt_WS, opt_WP = design_point
            plt.plot(opt_WS, opt_WP, marker='*', color='red', markersize=15, 
                     label=f'Design Point (W/S={opt_WS:.1f}, W/P={opt_WP:.3f})')
        
        plt.xlabel('Wing Loading (W/S)')
        plt.ylabel('Power Loading (W/P)')
        plt.title('Constraint / Matching Diagram')
        plt.legend(loc='lower left', fontsize='small')
        plt.grid()
        plt.ylim(0, 1)

        figures_dir = os.path.join(os.path.dirname(__file__), 'Class_I_Figures')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, 'matching_diagram.png'))
        plt.show()