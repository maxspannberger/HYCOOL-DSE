import numpy as np
import parameters as param
from Wing_Design import WingDesign
from Drag_Polar import DragPolarEstimator
from objects import Atmosphere, MatchingDiagram

# Extract necessary sizing variables from parameters file
PAYLOAD_KG = param.weight_parameters["payload_kg"]
C_OE       = param.weight_parameters["C_OE_guess"]

# ==============================================================================
# EXACT FUEL CALCULATION (Per Roskam Class I Equations)
# ==============================================================================
# 1. Product of all nominal mission phases (including loiter reserves)
# M_ff = (                                        # fuel fraction
#     param.mass_fractions["Mf_1"] * param.mass_fractions["Mf_2"] *
#     param.mass_fractions["Mf_3"] * param.mass_fractions["Mf_4"] *
#     param.mass_fractions["Mf_5"] * param.mass_fractions["Mf_7"] *
#     param.mass_fractions["Mf_8"] * param.mass_fractions["Mf_6"]
# )

M_ff = param.calculate_mass_fraction()
print(M_ff)

def run_class_1_sizing():
    """Iteratively converges MTOW based on fuel and payload requirements."""
    MTOW = param.flight_parameters["MTOW"] 
    tolerance = 0.01
    error = 1e6
    iteration = 1
    
    print("\n" + "="*50)
    print("  STARTING CLASS I WEIGHT ITERATION (KEROSENE)")
    print("="*50)
    
    while error > tolerance:
        # Calculate constituent weights
        W_OE = C_OE * MTOW
        W_F = (1 - M_ff) * MTOW
        
        # Calculate new take-off weight
        MTOW_calc = W_OE + W_F + PAYLOAD_KG
        
        # Determine convergence error
        error = abs(MTOW - MTOW_calc)
        MTOW = MTOW_calc
        iteration += 1
        
        # Breakout to prevent infinite loops
        if iteration > 1000:
            print("WARNING: Sizing failed to converge.")
            break

    print("\n" + "="*50)
    print("             ITERATION CONVERGED!")
    print("="*50)
    print(f"Final MTOW:    {MTOW:.2f} kg")
    print(f"Final OEW:     {W_OE:.2f} kg")
    print(f"Total fuel (including reserves):     {W_F:.2f} kg")
    print("=" * 50 + "\n")
    return MTOW

if __name__ == "__main__":
    
    # 1. Converge Weight
    converged_MTOW = run_class_1_sizing()
    param.flight_parameters["MTOW"] = converged_MTOW
    
    # 2. Match Performance (Generate Diagram)
    print(">>> OPTIMIZING MATCHING DIAGRAM...")
    atm = Atmosphere(param.flight_parameters["Cruise_altitude"])
    
    # Pass the perfectly converged MTOW into the Diagram for accurate scaling
    diagram = MatchingDiagram(MTOW=converged_MTOW)
    
    # Calculate performance curves and automatically extract the design point
    W_P_Curves, W_S_Curves = diagram.calculate_matching(atm)
    optimal_W_S, optimal_W_P = diagram.get_design_point(W_P_Curves, W_S_Curves)
    
    # Calculate Engine Power Requirement (Convert MTOW to Newtons, divide by N/W)
    total_peak_power_kw = ((converged_MTOW * 9.80665) / optimal_W_P) / 1000.0
    power_per_engine_kw = total_peak_power_kw / param.propulsion_parameters["Ne"]
    
    print(f"AUTOMATED DESIGN POINT FOUND:")
    print(f" -> Optimum Wing Loading (W/S):  {optimal_W_S:.0f} N/m²")
    print(f" -> Optimum Power Loading (W/P): {optimal_W_P:.4f} N/W")
    print(f" -> Total Peak Power Req:        {total_peak_power_kw:.2f} kW")
    print(f" -> Peak Power per Engine:       {power_per_engine_kw:.2f} kW\n")
    
    # 3. Size Wing Geometry using optimal W/S
    print(">>> CALCULATING WING GEOMETRY...")
    wing = WingDesign(
        W = converged_MTOW, 
        w = optimal_W_S, 
        A = param.aerodynamic_parameters["A"], 
        M_cr = param.flight_parameters["MCR"]
    )
    print(wing)
    
    # 4. Generate Performance Plots
    print("\n>>> GENERATING MATCHING DIAGRAM PLOT (Close window to continue)...")
    diagram.generate_diagram(W_P_Curves, W_S_Curves, design_point=(optimal_W_S, optimal_W_P))
    
    print("\n>>> GENERATING DRAG POLAR (Close plot window to finish)...")
    my_aircraft = DragPolarEstimator(
        W_kg=converged_MTOW, 
        S=wing.S, 
        b=wing.b, 
        Cd0=param.aerodynamic_parameters["CD0"], 
        e=param.aerodynamic_parameters["e"],
        rho=atm.density
    )
    V_lst = np.linspace(120, 400, 100)
    my_aircraft.calculate_performance(V_lst)
    my_aircraft.plot_results()