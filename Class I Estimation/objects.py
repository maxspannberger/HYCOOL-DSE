import numpy as np
import math
import os
import matplotlib.pyplot as plt
from parameters import lift_coefficients, propulsion_parameters, aerodynamic_parameters, flight_parameters, beta_dict

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
            
        self.temperature += self.isa_offset
        self.density = self.pressure / (self.R * self.temperature)
        self.speed_of_sound = math.sqrt(self.GAMMA * self.R * self.temperature)
        self.dynamic_viscosity = (self.MU_REF * ((self.temperature / self.T_REF) ** 1.5) * ((self.T_REF + self.S) / (self.temperature + self.S)))
        self.kinematic_viscosity = self.dynamic_viscosity / self.density

class MatchingDiagram:
    def __init__(self, Vs0=55.0, Vapp=60.0, LFL=1800, c=0, G=0.0024, TO_field_length=1000):
        self.lift_coefficients = lift_coefficients
        self.propulsion_parameters = propulsion_parameters
        self.aerodynamic_parameters = aerodynamic_parameters
        self.flight_parameters = flight_parameters

        self.Ne = self.propulsion_parameters["Ne"]  
        self.A = self.aerodynamic_parameters["A"]  
        self.e = self.aerodynamic_parameters["e"]  
        self.CD0 = self.aerodynamic_parameters["CD0"]  
        self.kP = self.propulsion_parameters["kP"]
        self.MTOW = self.flight_parameters["MTOW"]  # This gets updated dynamically by mainWeight.py!
        self.MCR = self.flight_parameters['MCR']  
        self.eta_prop = self.propulsion_parameters["eta_prop"]  
        self.beta = beta_dict

        self.gamma = Atmosphere.GAMMA
        self.R = Atmosphere.R
        self.g = Atmosphere.G

        self.CLFL = 0.45 
        self.h2 = 11 
        self.TO = 1800 
        self.kT = 0.85 

        self.density_SLS = Atmosphere(0).density  
        self.density_cruise = Atmosphere(self.flight_parameters['Cruise_altitude']).density
        
        self.Vs0 = Vs0 
        self.Vapp = Vapp 
        self.LFL = LFL 
        
        self.V_cr = self.MCR * Atmosphere(self.flight_parameters['Cruise_altitude']).speed_of_sound 
        self.c = c 
        self.G = G 
        self.LTO = self.flight_parameters['TO_field_length'] 
        self.sigma = self.density_cruise / Atmosphere(0).density 
        self.Dp = 4.0 

        # Landing Requirements (No hardcoded weights!)
        self.S_land = 1800 
        self.f_land = 1.67  
        self.h_land = 15.3  
        self.a_g = 0.4  

        self.g_climb = 0.024  
        self.W_S = np.linspace(10, 800, 1000) # [kg/m^2]

    def calculate_matching(self, atm: Atmosphere):
        # Convert W_S to Newtons for physical drag equations
        W_S_N = self.W_S * self.g 
        
        # 1. Cruise Speed 
        W_P_cruise_NW = (self.eta_prop / self.beta['beta_cruise']) / (((self.CD0 * 0.5 * self.density_cruise * np.power(self.V_cr, 3))/(self.beta['beta_cruise'] * W_S_N)) + ((self.beta['beta_cruise'] * W_S_N) / (np.pi * self.A * self.e * 0.5 * self.density_cruise * self.V_cr)))
        W_P_cruise = W_P_cruise_NW * (1000.0 / self.g) # Convert N/W to kg/kW

        # 2. Take-off Distance 
        self.CL2 = 0.694 * self.lift_coefficients['CL_max_TO']
        W_P_TO_NW = (1 / (1.15 * np.sqrt((self.Ne / (self.Ne - 1)) * (W_S_N / (self.TO * self.density_cruise * self.kT * self.g * self.A * self.e))) + (self.Ne / (self.Ne - 1)) * (4 * self.h2 / self.LTO))) * np.sqrt((self.CL2 / W_S_N) * (self.density_SLS / 2))
        W_P_TO = W_P_TO_NW * (1000.0 / self.g)

        # 3. Landing Distance (Using beta_landing fraction instead of hardcoded numbers!)
        landing_weight_fraction = self.beta['beta_landing']
        W_S_land_N = (1 / landing_weight_fraction) * ((self.S_land/(self.f_land * self.h_land)) - 10) * ((self.h_land * atm.density * self.g * self.lift_coefficients['CL_max_L'])/(1.52/self.a_g + 1.69))
        W_S_land = W_S_land_N / self.g # Convert N/m^2 back to kg/m^2

        # 4. Minimum Speed 
        W_S_min_N = (1 / self.beta['beta_landing']) * 0.5 * self.density_SLS * self.lift_coefficients['CL_max_L'] * np.square(self.Vapp / 1.23)
        W_S_min = W_S_min_N / self.g # Convert N/m^2 back to kg/m^2
        
        # 5. Climb Gradient 
        CD = self.CD0 + self.CL2 / (np.pi * self.A * self.e) 
        CL = self.CL2
        W_P_climb_NW = ((self.Ne - 1)/self.Ne) * self.eta_prop * (1 / self.beta['beta_climb']) * (1 / (self.g_climb + (CD / CL))) * np.sqrt((self.density_cruise / 2)*(CL / (self.beta['beta_climb'] * W_S_N)))  
        W_P_climb = W_P_climb_NW * (1000.0 / self.g)

        W_P_Curves = {
                "Cruise Speed Requirement": W_P_cruise,
                "Take-off Distance Requirement": W_P_TO,
                "Climb Gradient Requirement": W_P_climb
        }
        W_S_Curves = {
                "Landing Distance Requirement": W_S_land,
                "Minimum Speed Requirement": W_S_min
        }
        return W_P_Curves, W_S_Curves
    
    def get_design_point(self, W_P_Curves, W_S_Curves):
        if any(isinstance(v, np.ndarray) and v.size > 1 for v in W_S_Curves.values()):
            W_P_Curves, W_S_Curves = W_S_Curves, W_P_Curves
            
        optimum_W_S = min([float(v) for v in W_S_Curves.values()])
        idx = (np.abs(self.W_S - optimum_W_S)).argmin()
        optimum_W_P = min(float(curve[idx]) for curve in W_P_Curves.values())
        return optimum_W_S, optimum_W_P
        
    def generate_diagram(self, W_P_Curves, W_S_Curves, design_point=None):
        plt.figure(figsize=(10, 6))
        for curve_name, curve_data in W_P_Curves.items():
            plt.plot(self.W_S, curve_data, label=curve_name)
    
        plt.vlines(W_S_Curves['Minimum Speed Requirement'], ymin=0, ymax=10.0, label='Minimum Speed Requirement', colors='red', linestyles='dashed')
        plt.vlines(W_S_Curves['Landing Distance Requirement'], ymin=0, ymax=10.0, label='Landing Distance Requirement', colors='purple', linestyles='dashed')
        
        if design_point:
            opt_WS, opt_WP = design_point
            plt.plot(opt_WS, opt_WP, marker='*', color='red', markersize=15, 
                     label=f'Design Point (W/S={opt_WS:.0f}, W/P={opt_WP:.2f})')
        
        plt.xlabel('Wing Loading (W/S) [kg/m²]')
        plt.ylabel('Power Loading (W/P) [kg/kW]')
        plt.title('HYCOOL Matching Diagram')
        plt.legend(loc='lower left', fontsize='small')
        plt.grid()
        plt.ylim(0, 10) 
        
        figures_dir = os.path.join(os.path.dirname(__file__), 'Class_I_Figures')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, 'matching_diagram.png'))
        plt.show()