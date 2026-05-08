import numpy as np
# cruising speed and altitude are the main parameters that influence the design of a wing
# S_to = k_to * W_to / T_to * (V_2 ** 2 / 2g + h_to)
# k_to = 2.2 for twin engine aircraft
# limitation in takeoff safety speed can be translated into a wing loading requirement:
# W_to / S = 0.5 * rho * V_2 ** 2 * C_L_max * (V_s / V_2) ** 2 where V_s / V_2 is mentioned in the regulations

# the quarter chord sweep angle can be assumed to be 0 for Mach 0.7
class WingDesign:
    """
    Class I estimation for a wing geometry based on MTOW, wing loading, and planform parameters.
    """
    def __init__(self, W=31000, w=500, b=30, lambda_=0.4, Lambda_LE_deg=0.0):
        """
        Initialize the wing parameters and calculate the geometry.
        
        Args:
            W (float): Maximum takeoff weight [kg]
            w (float): Wing loading [kg/m^2]
            b (float): Wingspan [m]
            lambda_ (float): Taper ratio [-]
            Lambda_LE_deg (float): Leading edge sweep angle [deg]
        """
        # Primary Inputs
        self.W = W                   
        self.w = w                   
        self.b = b                   
        self.lambda_ = lambda_       
        self.Lambda_LE_deg = Lambda_LE_deg 
        
        # Convert sweep to radians for trigonometric functions
        self.Lambda_LE = np.deg2rad(self.Lambda_LE_deg)
        
        # Execute the geometric calculations
        self._calculate_wing_geometry()

    def _calculate_wing_geometry(self):
        """Calculates derived wing parameters."""
        # 1. Area and Aspect Ratio
        self.S = self.W / self.w                         # [m^2] Wing area
        self.A = self.b ** 2 / self.S                    # [-] Aspect ratio
        
        # 2. Chords
        self.C_r = 2 * self.S / ((1 + self.lambda_) * self.b)  # [m] Root chord
        self.C_t = self.lambda_ * self.C_r                     # [m] Tip chord
        
        # 3. Sweep
        # Quarter chord sweep angle [rad]
        self.Lambda_c4 = np.atan(np.tan(self.Lambda_LE) + self.C_r * (self.lambda_ - 1) / (2 * self.b)) 
        self.Lambda_c4_deg = np.rad2deg(self.Lambda_c4)        # [deg]
        
        # 4. Mean Aerodynamic & Geometric Chords
        self.MAC = (2 / 3) * self.C_r * (1 + self.lambda_ + self.lambda_ ** 2) / (1 + self.lambda_) # [m]
        
        # Location of the mean geometric chord
        self.y_MGC = (self.b / 6) * (1 + 2 * self.lambda_) / (1 + self.lambda_)                     # [m]
        self.x_MGC = np.tan(self.Lambda_LE) * self.y_MGC                                            # [m]

    def update_and_check(self, W=None, w=None, b=None, lambda_=None, Lambda_LE_deg=None):
        """
        Updates wing parameters and checks if they are identical to the old ones.
        Returns True if parameters remained the same, False if they changed.
        """
        old_params = (self.W, self.w, self.b, self.lambda_, self.Lambda_LE_deg)
        
        self.W = W if W is not None else self.W
        self.w = w if w is not None else self.w
        self.b = b if b is not None else self.b
        self.lambda_ = lambda_ if lambda_ is not None else self.lambda_
        self.Lambda_LE_deg = Lambda_LE_deg if Lambda_LE_deg is not None else self.Lambda_LE_deg
        
        new_params = (self.W, self.w, self.b, self.lambda_, self.Lambda_LE_deg)
        
        if old_params == new_params:
            return True
        
        # Recalculate if parameters changed
        self.Lambda_LE = np.deg2rad(self.Lambda_LE_deg)
        self._calculate_wing_geometry()
        return False

    def __repr__(self):
        """Provides a clean printed summary of the wing design."""
        return (f"--- Wing Class I Geometry ---\n"
                f"W (MTOW)      = {self.W} kg\n"
                f"S (Area)      = {self.S:.2f} m^2\n"
                f"b (Span)      = {self.b:.2f} m\n"
                f"Aspect Ratio  = {self.A:.2f}\n"
                f"Taper Ratio   = {self.lambda_:.2f}\n"
                f"C_r (Root)    = {self.C_r:.2f} m\n"
                f"C_t (Tip)     = {self.C_t:.2f} m\n"
                f"MAC           = {self.MAC:.2f} m\n"
                f"Sweep (LE)    = {self.Lambda_LE_deg:.2f} deg\n"
                f"Sweep (c/4)   = {self.Lambda_c4_deg:.2f} deg\n"
                f"x_MGC         = {self.x_MGC:.2f} m\n"
                f"y_MGC         = {self.y_MGC:.2f} m")
    
wing1 = WingDesign()
print(wing1)