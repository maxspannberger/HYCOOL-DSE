from pathlib import Path
import json
import csv


def collect_final_geometry(cfg, result) -> dict:
    """
    Collect the final aircraft geometry after the Class II iteration.

    This function does not change any calculations. It only extracts values
    from the converged ClassIIResult and AircraftConfig.
    """

    if not result.iteration_log:
        raise ValueError("No iteration log found. Run run_class_ii() first.")

    final = result.iteration_log[-1]

    tank_volume = result.W_fuel / cfg.rho_LH2_eff if cfg.rho_LH2_eff > 0 else 0.0

    geometry = {
        "Masses": {
            "MTOW_kg": result.MTOW,
            "MZFW_kg": result.MZFW,
            "OEW_kg": result.OEW,
            "fuel_mass_kg": result.W_fuel,
            "payload_kg": result.W_payload,
            "fixed_equipment_kg": result.W_fixed,
        },

        "Wing": {
            "S_ref_m2": final["S_ref_m2"],
            "span_m": final["b_m"],
            "aspect_ratio": cfg.AR,
            "root_chord_m": final["c_root_m"],
            "tip_chord_m": final["c_tip_m"],
            "MAC_m": final["MAC_m"],
            "taper_ratio": result.Wing_taper,
            "sweep_half_rad": cfg.sweep_half,
            "sweep_half_deg": cfg.sweep_half * 180 / 3.141592653589793,
            "sweep_tc_rad": cfg.sweep_tc,
            "sweep_tc_deg": cfg.sweep_tc * 180 / 3.141592653589793,
            "thickness_to_chord_root": cfg.tc_root,
            "thickness_to_chord_mean": cfg.tc_mean,
            "root_thickness_m": cfg.tc_root * final["c_root_m"],
            "LEMAC_m_from_nose": cfg.LEMAC,
        },

        "Horizontal_tail": {
            "S_h_m2": result.tail_rechecked.S_h,
            "span_h_m": result.tail_rechecked.b_h,
            "MAC_h_m": cfg.MAC_h,
            "tail_arm_l_h_m": cfg.l_h,
            "aspect_ratio_h": cfg.AR_h,
            "V_h": result.tail_rechecked.V_h,
            "elevator_area_m2": result.tail_rechecked.S_elevator,
            "Se_Sh": result.tail_rechecked.Se_Sh,
            "driver": result.tail_rechecked.S_h_driver,
        },

        "Vertical_tail": {
            "S_v_m2": result.tail_rechecked.S_v,
            "span_v_m": result.tail_rechecked.b_v,
            "MAC_v_m": cfg.MAC_v,
            "tail_arm_l_v_m": cfg.l_v,
            "aspect_ratio_v": cfg.AR_v,
            "V_v": result.tail_rechecked.V_v,
            "rudder_area_m2": result.tail_rechecked.S_rudder,
            "Sr_Sv": result.tail_rechecked.Sr_Sv,
            "driver": result.tail_rechecked.S_v_driver,
        },

        "Fuselage": {
            "length_m": final["l_f_m"],
            "diameter_equivalent_m": final["d_f_m"],
            "width_m": cfg.b_f,
            "height_m": cfg.h_f,
            "wetted_area_m2": final["S_wet_f_m2"],
            "LEMAC_m_from_nose": cfg.LEMAC,
        },

        "LH2_tank": {
            "fuel_mass_kg": result.W_fuel,
            "effective_LH2_density_kg_m3": cfg.rho_LH2_eff,
            "volume_m3": tank_volume,
            "radius_m": final["r_tank_m"],
            "diameter_m": final["d_tank_m"],
            "length_m": final["L_tank_m"],
            "hump_back": cfg.hump_back,
            "extra_wetted_area_hump_m2": final["S_wet_hump_m2"],
            "tank_weight_kg": result.weight.W_h2_tank,
        },

        "Aerodynamics": {
            "cruise_CL": result.CL_cruise,
            "cruise_L_over_D": result.L_over_D,
            "CD0": result.drag.CD0,
            "CD_total": result.drag.CD_total,
            "oswald_e": result.drag.e,
        },

        "Power": {
            "P_cruise_kW": result.P_cruise_KW,
            "P_climb_kW": result.P_climb_KW,
            "P_reserve_kW": result.P_reserve_KW,
            "P_TO_kW": result.P_TO_KW,
            "P_TO_OEI_kW": result.P_TO_OEI_KW,
            "static_thrust_per_engine_kN": result.power.T_static_per_engine / 1000,
        },

        "Distances_mac": {
            "distance_le_mac_to_cg": result.distance_le_mac_to_cg,
            "distance_le_mac_to_turbine": result.distance_le_mac_to_turbine,
            "distance_le_root_to_le_mac": result.distance_le_root_to_le_mac,
        },
    }

    return geometry


def print_final_geometry(cfg, result) -> None:
    """
    Simple readable console printout.
    """
    geometry = collect_final_geometry(cfg, result)

    print("\n========== FINAL AIRCRAFT GEOMETRY ==========\n")

    for group, values in geometry.items():
        print(f"\n[{group}]")
        for key, value in values.items():
            if isinstance(value, float):
                print(f"{key:35s}: {value:.4f}")
            else:
                print(f"{key:35s}: {value}")


def export_final_geometry(cfg, result, output_dir="outputs") -> dict:
    """
    Export final geometry to JSON and CSV.
    """
    geometry = collect_final_geometry(cfg, result)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "final_aircraft_geometry.json"
    csv_path = output_dir / "final_aircraft_geometry.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(geometry, f, indent=4)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "parameter", "value"])

        for group, values in geometry.items():
            for key, value in values.items():
                writer.writerow([group, key, value])

    return {
        "json": json_path,
        "csv": csv_path,
    }