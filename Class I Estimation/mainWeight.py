import numpy as np
import parameters as param
from Wing_Design import WingDesign
from Drag_Polar import DragPolarEstimator
from objects import Atmosphere, MatchingDiagram

PAYLOAD_KG = param.weight_parameters["payload_kg"]
C_OE       = param.weight_parameters["C_OE_guess"]

# ==============================================================================
# EXACT FUEL CALCULATION (Equations 2.13, 2.14, 2.15)
# ==============================================================================
M_end_M_TO = (
    param.mass_fractions["Mf_1"] * param.mass_fractions["Mf_2"] *
    param.mass_fractions["Mf_3"] * param.mass_fractions["Mf_4"] *
    param.mass_fractions["Mf_5"] * param.mass_fractions["Mf_7"] *
    param.mass_fractions["Mf_8"]
)

M_ff_m = 1.0 - M_end_M_TO
M_loiter_M_end = param.mass_fractions["Mf_6"]
M_ff_res = (1.0 - M_loiter_M_end) * M_end_M_TO + (param.weight_parameters["contingency_margin"] * M_ff_m)
M_ff = M_ff_m + M_ff_res

def run_class_1_sizing():
    MTOW = param.flight_parameters["MTOW"] 
    tolerance = 0.1
    error = 1e6
    iteration = 1
    
    print("\n" + "="*50)
    print("  STARTING CLASS I WEIGHT ITERATION (KEROSENE)")
    print("="*50)
    
    while error > tolerance:
        W_OE = C_OE * MTOW
        W_F_used = M_ff_m * MTOW
        W_F_res_kg = M_ff_res * MTOW
        W_F = W_F_used + W_F_res_kg
        MTOW_calc = W_OE + W_F + PAYLOAD_KG
        
        error = abs(MTOW - MTOW_calc)
        MTOW = MTOW_calc
        iteration += 1
        
        if iteration > 1000:
            print("WARNING: Sizing failed to converge.")
            break

    print("\n" + "="*50)
    print("             ITERATION CONVERGED!")
    print("="*50)
    print(f"Final MTOW:    {MTOW:.2f} kg")
    print(f"Final OEW:     {W_OE:.2f} kg")
    print(f"Fuel Used:     {W_F_used:.2f} kg")
    print(f"Reserve Fuel:  {W_F_res_kg:.2f} kg")
    print("=" * 50 + "\n")
    return MTOW

if __name__ == "__main__":
    
    # 1. Size Weight
    converged_MTOW = run_class_1_sizing()
    
    # Update global parameter dictionary strictly for reference
    param.flight_parameters["MTOW"] = converged_MTOW
    
    # 2. Match Performance
    print(">>> OPTIMIZING MATCHING DIAGRAM...")
    atm = Atmosphere(param.flight_parameters["Cruise_altitude"])
    
    # Explicitly pass the converged weight into the MatchingDiagram!
    diagram = MatchingDiagram(MTOW=converged_MTOW)
    
    W_P_Curves, W_S_Curves = diagram.calculate_matching(atm)
    optimal_W_S, optimal_W_P = diagram.get_design_point(W_P_Curves, W_S_Curves)
    
    # PEAK POWER CALCULATION WITH N/W
    total_peak_power_kw = ((converged_MTOW * 9.80665) / optimal_W_P) / 1000.0
    power_per_engine_kw = total_peak_power_kw / param.propulsion_parameters["Ne"]
    
    print(f"AUTOMATED DESIGN POINT FOUND:")
    print(f" -> Optimum Wing Loading (W/S):  {optimal_W_S:.0f} N/m²")
    print(f" -> Optimum Power Loading (W/P): {optimal_W_P:.4f} N/W")
    print(f" -> Total Peak Power Req:        {total_peak_power_kw:.2f} kW")
    print(f" -> Peak Power per Engine:       {power_per_engine_kw:.2f} kW\n")
    
    # 3. Size Wing Geometry
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
    V_lst = np.linspace(20, 180, 100)
    my_aircraft.calculate_performance(V_lst)
    my_aircraft.plot_results()