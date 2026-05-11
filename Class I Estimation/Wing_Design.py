import numpy as np

class WingDesign:
    """Class I estimation for wing geometry."""
    
    def __init__(self, W, w, A, M_cr):
        """
        Initializes the wing design parameters.
        W: Mass in kg
        w: Wing Loading in N/m^2
        A: Aspect Ratio
        M_cr: Cruise Mach
        """
        self.W = W
        self.w = w
        self.A = A
        self.M_cr = M_cr
        self._calculate_wing_geometry()

    def _calculate_wing_geometry(self):
        """Calculates precise geometric dimensions of the wing planform."""
        
        # W is in kg. w is in N/m^2. Multiply W by gravity to get N to calculate area (S).
        self.S = (self.W * 9.80665) / self.w
        self.b = np.sqrt(self.A * self.S)
        
        # Quarter Chord Sweep Angle
        self.Lambda_c4_rad = np.arccos(1.16 / (self.M_cr + 0.5))
        self.Lambda_c4_deg = np.rad2deg(self.Lambda_c4_rad)
        
        # Taper Ratio
        self.lambda_ = 0.2 * (2 - self.Lambda_c4_rad)
        
        # Leading Edge Sweep Angle
        tan_LE = np.tan(self.Lambda_c4_rad) + (1 - self.lambda_) / (self.A * (1 + self.lambda_))
        self.Lambda_LE_rad = np.arctan(tan_LE)
        self.Lambda_LE_deg = np.rad2deg(self.Lambda_LE_rad)
        
        # Half Chord Sweep Angle
        tan_c2 = np.tan(self.Lambda_c4_rad) - (1 - self.lambda_) / (self.A * (1 + self.lambda_))
        self.Lambda_c2_rad = np.arctan(tan_c2)
        self.Lambda_c2_deg = np.rad2deg(self.Lambda_c2_rad)
        
        # Root and Tip Chords
        self.C_r = 2 * self.S / ((1 + self.lambda_) * self.b)
        self.C_t = self.lambda_ * self.C_r
        
        # Mean Aerodynamic Chord
        self.MAC = (2 / 3) * self.C_r * (1 + self.lambda_ + self.lambda_ ** 2) / (1 + self.lambda_)
        
        # Mean Geometric Chord Location
        self.y_MGC = (self.b / 6) * (1 + 2 * self.lambda_) / (1 + self.lambda_)
        self.x_MGC = np.tan(self.Lambda_LE_rad) * self.y_MGC

    def update_and_check(self, W=None, w=None, A=None, M_cr=None):
        """Allows parameters to be updated without re-instantiating the object."""
        old_params = (self.W, self.w, self.A, self.M_cr)
        
        self.W = W if W is not None else self.W
        self.w = w if w is not None else self.w
        self.A = A if A is not None else self.A
        self.M_cr = M_cr if M_cr is not None else self.M_cr
        
        new_params = (self.W, self.w, self.A, self.M_cr)
        
        if old_params == new_params:
            return True
            
        self._calculate_wing_geometry()
        return False

    def __repr__(self):
        return (f"--- Wing Class I Geometry ---\n"
                f"W (MTOW)      = {self.W:.2f} kg\n"
                f"S (Area)      = {self.S:.2f} m²\n"
                f"b (Span)      = {self.b:.2f} m\n"
                f"Aspect Ratio  = {self.A:.2f}\n"
                f"Taper Ratio   = {self.lambda_:.2f}\n"
                f"C_r (Root)    = {self.C_r:.2f} m\n"
                f"C_t (Tip)     = {self.C_t:.2f} m\n"
                f"MAC           = {self.MAC:.2f} m\n"
                f"Sweep (LE)    = {self.Lambda_LE_deg:.2f} deg\n"
                f"Sweep (c/4)   = {self.Lambda_c4_deg:.2f} deg\n"
                f"Sweep (c/2)   = {self.Lambda_c2_deg:.2f} deg\n"
                f"x_MGC         = {self.x_MGC:.2f} m\n"
                f"y_MGC         = {self.y_MGC:.2f} m")