import numpy as np
from dataclasses import dataclass, replace
import sys
from pathlib import Path
import json

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from WeightEstimations.mainClassII   import run_class_ii
from WeightEstimations.Aircraft_Config   import AircraftConfig, default_q400_hycool
from General.component_parameters import component_params as comp_params
from WeightEstimations.ISA import isa

_optimal_cl_mach_cache: dict | None = None
_optimal_cl_mach_path = Path(__file__).resolve().parent / "outputs" / "optimal_cl_mach_cache.json"


def apply_optimal_cl_mach(cfg: AircraftConfig, best_row: dict) -> AircraftConfig:
    """Return a copy of cfg with the optimal cruise Mach applied."""
    from dataclasses import replace
    return replace(cfg, M_cruise=best_row["M_cruise"])


def get_optimal_cl_mach(cfg: AircraftConfig, force_recompute: bool = False) -> dict:
    """Return the cached optimal Mach result, computing it only once unless forced."""
    global _optimal_cl_mach_cache
    if _optimal_cl_mach_cache is None and not force_recompute:
        if _optimal_cl_mach_path.exists():
            with open(_optimal_cl_mach_path, "r", encoding="utf-8") as f:
                _optimal_cl_mach_cache = json.load(f)

    if _optimal_cl_mach_cache is None or force_recompute:
        _optimal_cl_mach_cache = find_optimal_cl_mach(cfg, force_recompute=force_recompute)
    return _optimal_cl_mach_cache.copy()


def find_optimal_cl_mach(cfg: AircraftConfig, force_recompute: bool = False) -> dict:
    # This function is a placeholder for the actual optimal CL and Mach calculation.
    # In a real implementation, this would involve more complex logic and possibly
    # iterative methods to find the optimal values based on the aircraft configuration.
    global _optimal_cl_mach_cache
    if _optimal_cl_mach_cache is not None and not force_recompute:
        return _optimal_cl_mach_cache.copy()

    from dataclasses import replace
    M_cruise=0.7
    sweep_rows=[]
    iterations=0
    T,p,rho=isa(cfg.altitude_cruise)
    a_cruise=np.sqrt(1.4*287.05*T) # speed of sound at cruise altitude
    while M_cruise>=0.66:
        factor=0.01
        cfg_updated = replace(cfg, M_cruise=M_cruise,V_cruise=M_cruise*a_cruise)
        result = run_class_ii(cfg_updated,comp=comp_params, tol=1.0, max_iter=100, verbose=False)
        value = cfg_updated.M_cruise*result.drag.CL_cruise/result.drag.CD_total
        CL_cruise = result.drag.CL_cruise
        
        sweep_rows.append({
            "value": value,
            "M_cruise": cfg_updated.M_cruise,
            "CL_cruise": CL_cruise,
            "CD_total": result.drag.CD_total,
            "t_cruise": result.mission.t_cruise,
            "m_LH2_cruise": result.mission.m_LH2_cruise,
            "m_LH2_climb": result.mission.m_LH2_climb,
            "m_LH2_taxi_TO": result.mission.m_LH2_TO_taxi,
            "MTOW": result.MTOW,
            "Wing Area": result.Wing_Area,
            "Stall Speed": cfg_updated.V_stall,
            "CL_max_TO": result.power.CL_max_TO,
            "CL_max_clean": cfg_updated.CL_max,
            "half_sweep": cfg_updated.sweep_half,
            "LE_sweep": cfg_updated.sweep_tc,
            "root_chord": cfg_updated.c_root,
            "span": cfg_updated.b,
            "taper": result.Wing_taper,
            "Aileron_Area_ratio": result.tail_rechecked.Sa_Sref,
        })

        iterations+=1
        print(f"Completed iteration {iterations} with M_cruise={M_cruise:.2f}, value={value:.6f}, CL_cruise={CL_cruise:.6f}, CD_total={result.drag.CD_total:.6f}, t_cruise={result.mission.t_cruise/60:.2f} min, m_LH2_cruise={result.mission.m_LH2_cruise:.2f} kg")
        M_cruise=M_cruise-factor

    reference_row = max(sweep_rows, key=lambda row: row["m_LH2_cruise"])
    reference_fuel = reference_row["m_LH2_cruise"]
    for row in sweep_rows:
        row["fuel_savings"] = reference_fuel - row["m_LH2_cruise"]

    best_row = max(sweep_rows, key=lambda row: row["value"])
    best_row["Altitude_cruise"] = cfg_updated.altitude_cruise
    best_row["Hydrogen Mass Needed"]=result.W_fuel
    _optimal_cl_mach_cache = best_row.copy()

    _optimal_cl_mach_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_optimal_cl_mach_path, "w", encoding="utf-8") as f:
        json.dump(_optimal_cl_mach_cache, f, indent=4)

    print(best_row)

    return best_row

if __name__ == "__main__":
    cfg = default_q400_hycool()
    Optcl1=get_optimal_cl_mach(cfg, force_recompute=True)
    cfg_2prop = replace(cfg, altitude_cruise=7620, V_cruise_EAS=140.938, V_dive=176.2133072)
    Optcl2=get_optimal_cl_mach(cfg_2prop, force_recompute=True)

    optimal=max(Optcl1["value"], Optcl2["value"])
    difference = abs(Optcl1["value"] - Optcl2["value"])
    print(f"Optimal value between altitude configurations: {optimal:.6f} (difference of {difference:.6f})")
    fuel1=Optcl1["Hydrogen Mass Needed"]
    fuel2=Optcl2["Hydrogen Mass Needed"]
    if optimal == Optcl1["value"]:
        optimal_mach = Optcl1["M_cruise"]
        optimal_altitude = Optcl1["Altitude_cruise"]
        print(f"Fuel needed at optimal point: {fuel1:.2f} kg (difference of {(fuel1-fuel2):.2f} kg between configs)")
    else:
        optimal_mach = Optcl2["M_cruise"]
        optimal_altitude = Optcl2["Altitude_cruise"]
        print(f"Fuel needed at optimal point: {fuel2:.2f} kg (difference of {(fuel1-fuel2):.2f} kg between configs) at altitude {optimal_altitude} with M_cruise={optimal_mach:.2f}")
    print(f"Optimal cruise Mach number with 4 props at depending on flight level: {optimal:.6f} at altitude {optimal_altitude} with M_cruise={optimal_mach:.2f}")