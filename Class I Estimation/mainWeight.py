import numpy as np
import parameters as param
from Wing_Design import WingDesign
import Drag_Polar
from objects import Atmosphere, MatchingDiagram

# ==============================================================================
# 1. EXTRACT CONSTANTS
# ==============================================================================
PAYLOAD_KG = param.weight_parameters["payload_kg"]
C_OE       = param.weight_parameters["C_OE_guess"]

# ==============================================================================
# 2. NOMINAL MISSION FRACTION (Equation 2.13)
# ==============================================================================
# Product of all phases EXCEPT loiter (which is a reserve phase)
M_ff = (
    param.mass_fractions["Mf_1"] * param.mass_fractions["Mf_2"] *
    param.mass_fractions["Mf_3"] * param.mass_fractions["Mf_4"] *
    param.mass_fractions["Mf_5"] * param.mass_fractions["Mf_7"] *
    param.mass_fractions["Mf_8"]
)

# ==============================================================================
# 3. CLASS I ITERATIVE SIZING LOOP
# ==============================================================================
def run_class_1_sizing():
    MTOW = param.flight_parameters["MTOW"] 
    tolerance = 0.1
    error = 1e6
    iteration = 1
    
    print("\n" + "="*50)
    print("  STARTING CLASS I WEIGHT ITERATION (KEROSENE)")
    print("="*50)
    print(f"Mission Fraction Product (M_ff): {M_ff:.4f}")
    
    while error > tolerance:
        # 1. Calculate Empty Weight
        W_OE = C_OE * MTOW
        
        # 2. Calculate Fuel Used (Equation 2.14)
        W_F_used = (1.0 - M_ff) * MTOW
        
        # 3. Calculate Reserves (Loiter + Contingency)
        # Weight at the start of loiter is approximately M_ff * MTOW
        W_loiter_start = M_ff * MTOW
        W_F_loiter = (1.0 - param.mass_fractions["Mf_6"]) * W_loiter_start
        W_F_contingency = param.weight_parameters["contingency_margin"] * W_F_used
        
        W_F_res = W_F_loiter + W_F_contingency
        
        # 4. Total Fuel Weight (Equation 2.15)
        W_F = W_F_used + W_F_res
        
        # 5. Calculate New MTOW
        MTOW_calc = W_OE + W_F + PAYLOAD_KG
        
        error = abs(MTOW - MTOW_calc)
        print(f"Iter {iteration:02d}: Calculated MTOW = {MTOW_calc:.2f} kg")
        
        MTOW = MTOW_calc
        iteration += 1
        
        if iteration > 1000:
            print("WARNING: Sizing failed to converge.")
            break

    print("\n" + "="*50)
    print("             ITERATION CONVERGED!")
    print("="*50)
    print(f"Final MTOW:     {MTOW:.2f} kg")
    print(f"Final OEW:      {W_OE:.2f} kg")
    print(f"Fuel Used:      {W_F_used:.2f} kg")
    print(f"Fuel Reserves:  {W_F_res:.2f} kg")
    print(f"Total Fuel:     {W_F:.2f} kg")
    print("=" * 50 + "\n")
    return MTOW

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    # A. Run Weight Loop 
    converged_MTOW = run_class_1_sizing()
    
    # B. Update Global MTOW parameter
    param.flight_parameters["MTOW"] = converged_MTOW
    
    # C. Run Matching Diagram FIRST to find optimal Wing Loading
    print(">>> OPTIMIZING MATCHING DIAGRAM...")
    atm = Atmosphere(param.flight_parameters["Cruise_altitude"])
    diagram = MatchingDiagram()
    
    W_P_Curves, W_S_Curves = diagram.calculate_matching(atm)
    optimal_W_S, optimal_W_P = diagram.get_design_point(W_P_Curves, W_S_Curves)
    
    print(f"AUTOMATED DESIGN POINT FOUND:")
    print(f" -> Optimum Wing Loading (W/S):  {optimal_W_S:.2f} kg/m^2")
    print(f" -> Optimum Power Loading (W/P): {optimal_W_P:.4f} kg/W\n")
    
    # D. Size the Wing using the newly found optimal W/S
    print(">>> CALCULATING WING GEOMETRY...")
    wing = WingDesign(
        W = converged_MTOW, 
        w = optimal_W_S, 
        A = param.aerodynamic_parameters["A"], 
        M_cr = param.flight_parameters["MCR"]
    )
    print(wing)
    
    # E. Display the Diagram
    print("\n>>> GENERATING MATCHING DIAGRAM PLOT (Close window to continue)...")
    diagram.generate_diagram(W_P_Curves, W_S_Curves, design_point=(optimal_W_S, optimal_W_P))
    
    # F. Display the Drag Polar
    print("\n>>> GENERATING DRAG POLAR (Close plot window to finish)...")
    Drag_Polar.plot_dat_thang(
        W = converged_MTOW, 
        rho = atm.density, 
        S = wing.S, 
        Cd0 = param.aerodynamic_parameters["CD0"], 
        e = param.aerodynamic_parameters["e"], 
        A = wing.A, 
        V_lst = Drag_Polar.V_lst
    )
    print("\n>>> BASELINE SIZING COMPLETE. <<<")