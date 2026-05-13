import numpy as np
import math
import os
import matplotlib.pyplot as plt
from parameters import lift_coefficients, propulsion_parameters, aerodynamic_parameters, flight_parameters, climb_gradients, beta_dict

class Atmosphere:
    """
    International Standard Atmosphere (ISA) model.
    Calculates atmospheric properties up to 20,000 meters.
    """
    G = 9.80665          # Gravity (m/s^2)
    R = 287.0528         # Specific gas constant for dry air (J/(kg·K))
    GAMMA = 1.4          # Specific heat ratio for air
    S = 110.4            # Sutherland's temperature (K)
    MU_REF = 1.716e-5    # Reference dynamic viscosity (kg/(m·s))
    T_REF = 273.15       # Reference temperature (K)

    def __init__(self, altitude_m, isa_offset_c=0.0):
        if altitude_m < 0:
            raise ValueError("Altitude must be zero or positive.")
            
        self.altitude = altitude_m
        self.isa_offset = isa_offset_c
        self._calculate_state()

    def _calculate_state(self):
        # --- Layer 0: Troposphere (0 to 11,000 m) ---
        if self.altitude <= 11000.0:
            Tb = 288.15
            Pb = 101325.0
            a = -0.0065
            self.temperature = Tb + a * self.altitude
            self.pressure = Pb * (self.temperature / Tb) ** (-self.G / (a * self.R))
            
        # --- Layer 1: Tropopause (11,000 m to 20,000 m) ---
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

class MatchingDiagram:
    """Calculates performance constraint curves and optimal W/S & W/P points."""
    
    def __init__(self, MTOW, Vs0=55.0, Vapp=60.0, LFL=1050, c=0, G=0.0024):
        
        self.MTOW = MTOW # dynamically passed from main script
        
        # Assign Parameter Dictionaries
        self.lift_coefficients = lift_coefficients
        self.propulsion_parameters = propulsion_parameters
        self.aerodynamic_parameters = aerodynamic_parameters
        self.flight_parameters = flight_parameters
        self.beta = beta_dict
        self.climb_gradients = climb_gradients

        # Core Aircraft Data
        self.Ne = self.propulsion_parameters["Ne"]  
        self.A = self.aerodynamic_parameters["A"]  
        self.e = self.aerodynamic_parameters["e"]  
        self.CD0 = self.aerodynamic_parameters["CD0"]  
        self.kP = self.propulsion_parameters["kP"]
        self.MCR = self.flight_parameters['MCR']  
        self.eta_prop = self.propulsion_parameters["eta_prop"]  

        # Constants
        self.gamma = Atmosphere.GAMMA
        self.R = Atmosphere.R
        self.g = Atmosphere.G

        # Take-off Requirements
        self.h2 = 11      # [m]
        self.TO = 1263.314    # Take-off field length [m]
        self.kT = 0.9     # Take-off thrust parameter [-]
        self.density_TO = Atmosphere(610).density 
        # self.alpha_p = 1.0
        
        # Sea Level Properties
        self.density_SLS = Atmosphere(0).density  
        
        # Atmospheric Cruise Properties
        self.density_cruise = Atmosphere(self.flight_parameters['Cruise_altitude']).density
        
        # Speed Requirements
        self.Vs0 = Vs0 
        self.Vapp = Vapp 
        self.LFL = 915 # [m], same as ATR72-600 at max landing weight 
        self.CLFL = 0.45 # taken from ADSEE, 
        
        # Cruise speed calculated using speed of sound
        self.V_cr = self.MCR * Atmosphere(self.flight_parameters['Cruise_altitude']).speed_of_sound 
        
        # Array for Plotting Wing Loading [N/m^2]
        self.W_S = np.linspace(100, 8000, 1000)

    def calculate_matching(self, atm: Atmosphere):
        
        # 1. Cruise Speed Requirement (ADSEE 154)
        W_P_cruise = self.eta_prop * ((self.density_cruise/self.density_SLS)**(3/4) / self.beta['beta_cruise']) * ((self.CD0 * 0.5 * self.density_cruise * (self.V_cr ** 3))/(self.beta['beta_cruise'] * self.W_S) + (self.beta['beta_cruise'] * self.W_S) / (np.pi * self.A * self.e * 0.5 * self.density_cruise * self.V_cr))**(-1)

        # 2. Take-off Distance Requirement (ADSEE 176)
        self.CL2 = 0.694 * self.lift_coefficients['CL_max_TO']
        W_P_TO = (1.15 * np.sqrt(self.Ne/(self.Ne - 1) * self.W_S / (self.TO * self.kT * self.density_TO * self.g * np.pi * self.A * self.e)) + self.Ne/(self.Ne - 1) * 4 * self.h2/self.TO)**(-1) * np.sqrt(self.CL2 / self.W_S * self.density_TO / 2)

        # 3. Landing Distance Requirement (ADSEE 152)
        W_S_land = (1 / self.beta['beta_landing']) * (self.LFL/self.CLFL) * (self.density_SLS * self.lift_coefficients['CL_max_L']) / 2

        # 4. Minimum Speed Requirement (ADSEE 166)
        W_S_min = (1 / self.beta['beta_landing']) * 0.5 * self.density_SLS * self.lift_coefficients['CL_max_L'] * np.square(self.Vapp / 1.23)
    
        # 5. Climb Gradient Requirement (OEI) (ADSEE 161)

        CL_climb_landing = self.lift_coefficients['CL_max_L']
        CL_climb_TO_LG_extended = self.lift_coefficients['CL_max_TO']
        CL_climb_TO_LG_retracted = self.lift_coefficients['CL_max_TO']
        CL_climb_approach = self.lift_coefficients['CL_max_L']

        CD_climb_landing = self.CD0 + CL_climb_landing**2 / (np.pi * self.A * self.e) 
        CD_climb_TO_LG_extended = self.CD0 + CL_climb_TO_LG_extended**2 / (np.pi * self.A * self.e) 
        CD_climb_TO_LG_retracted = self.CD0 + CL_climb_TO_LG_retracted**2 / (np.pi * self.A * self.e) 
        CD_climb_approach = self.CD0 + CL_climb_approach**2 / (np.pi * self.A * self.e) 

        # Climb Gradient Requirement for Landing CS 25.119
        W_P_climb_landing = ((self.Ne - 1)/self.Ne) * self.eta_prop * (1 / self.beta['beta_climb']) * (1 / (self.climb_gradients['Landing'] + (CD_climb_landing / CL_climb_landing))) * np.sqrt((self.density_TO / 2)*(CL_climb_landing / (self.beta['beta_climb'] * self.W_S)))  
        W_P_climb_TO_LG_extended = ((self.Ne - 1)/self.Ne) * self.eta_prop * (1 / self.beta['beta_climb']) * (1 / (self.climb_gradients['Take-Off_LG_Extended'] + (CD_climb_TO_LG_extended / CL_climb_TO_LG_extended))) * np.sqrt((self.density_TO / 2)*(CL_climb_TO_LG_extended / (self.beta['beta_climb'] * self.W_S)))
        W_P_climb_TO_LG_retracted = ((self.Ne - 1)/self.Ne) * self.eta_prop * (1 / self.beta['beta_climb']) * (1 / (self.climb_gradients['Take-Off_LG_Retracted'] + (CD_climb_TO_LG_retracted / CL_climb_TO_LG_retracted))) * np.sqrt((self.density_TO / 2)*(CL_climb_TO_LG_retracted / (self.beta['beta_climb'] * self.W_S)))
        W_P_climb_approach = ((self.Ne - 1)/self.Ne) * self.eta_prop * (1 / self.beta['beta_climb']) * (1 / (self.climb_gradients['Approach'] + (CD_climb_approach / CL_climb_approach))) * np.sqrt((self.density_TO / 2)*(CL_climb_approach / (self.beta['beta_climb'] * self.W_S)))

        # Store constraints
        W_P_Curves = {
                "Cruise Speed Requirement": W_P_cruise,
                "Take-off Distance Requirement": W_P_TO,
                "Climb Gradient Requirement Landing Configuration": W_P_climb_landing,
                "Climb Gradient Requirement TO Configuration with LG Extended": W_P_climb_TO_LG_extended,
                "Climb Gradient Requirement TO Configuration with LG Retracted": W_P_climb_TO_LG_retracted,
                "Climb Gradient Requirement Approach Configuration": W_P_climb_approach
        }

        W_S_Curves = {
                "Landing Distance Requirement": W_S_land,
                "Minimum Speed Requirement": W_S_min
        }
        return W_P_Curves, W_S_Curves
    
    def get_design_point(self, W_P_Curves, W_S_Curves):
        """Mathematically scans arrays to find the highest legal W/S and W/P."""
        # Safety catch in case arguments are swapped
        if any(isinstance(v, np.ndarray) and v.size > 1 for v in W_S_Curves.values()):
            W_P_Curves, W_S_Curves = W_S_Curves, W_P_Curves
            
        # The strictest boundary constraint determines Wing Loading
        optimum_W_S = min([float(v) for v in W_S_Curves.values()])
        idx = (np.abs(self.W_S - optimum_W_S)).argmin()
        
        # The lowest thrust curve at that W/S determines Engine size
        optimum_W_P = min(float(curve[idx]) for curve in W_P_Curves.values())
        return optimum_W_S, optimum_W_P
        
    def generate_diagram(self, W_P_Curves, W_S_Curves, design_point=None):
        """Generates visual representation of the Design Space."""
        plt.figure(figsize=(10, 6))
        
        # Plot continuous constraint curves
        for curve_name, curve_data in W_P_Curves.items():
            plt.plot(self.W_S, curve_data, label=curve_name)
        
        # Plot hard vertical boundaries. Ymax restricted to 0.2 per user setup.
        plt.vlines(W_S_Curves['Minimum Speed Requirement'], ymin=0, ymax=0.2, label='Minimum Speed Requirement', colors='black', linestyles='dashed')
        plt.vlines(W_S_Curves['Landing Distance Requirement'], ymin=0, ymax=0.2, label='Landing Distance Requirement', colors='purple', linestyles='dashed')
        
        # Mark Optimal Design Point
        if design_point:
            opt_WS, opt_WP = design_point
            plt.plot(opt_WS, opt_WP, marker='*', color='red', markersize=15, 
                     label=f'Design Point (W/S={opt_WS:.0f}, W/P={opt_WP:.4f})')
        
        plt.xlabel('Wing Loading (W/S) [N/m²]')
        plt.ylabel('Power Loading (W/P) [N/W]')
        plt.title('HYCOOL Matching Diagram')
        plt.legend(loc='upper right', fontsize='small')  
        plt.grid()
        plt.ylim(0, 0.2) 
        
        figures_dir = os.path.join(os.path.dirname(__file__), 'Class_I_Figures')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, 'matching_diagram.png'))
        plt.show()