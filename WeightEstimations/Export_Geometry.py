from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json


# Source labels used in the exported source map:
# - "cfg": value comes directly from AircraftConfig, so it is an input or assumption.
# - "result": value comes directly from ClassIIResult, so it is a final integrated Class II output.
# - "final": value comes from result.iteration_log[-1], so it is the final iteration geometry/log value.
# - "derived": value is calculated inside this export file from cfg, result, final, or tail_rechecked.
# - "tail_rechecked": value comes from result.tail_rechecked, after the final power/thrust recheck.
# - "drag": value comes from result.drag.
# - "power": value comes from result.power.
# - "weight": value comes from result.weight.


SOURCE_MAP: dict[str, dict[str, str]] = {
    "Masses": {
        "MTOW_kg": "result.MTOW",
        "MZFW_kg": "result.MZFW",
        "OEW_kg": "result.OEW",
        "empty_weight_without_fixed_kg": "result.W_empty",
        "fuel_mass_kg": "result.W_fuel",
        "payload_kg": "result.W_payload",
        "fixed_equipment_kg": "result.W_fixed",
        "propulsion_system_with_tank_kg": "result.W_prop",
    },

    "Wing": {
        "S_ref_m2": "final['S_ref_m2']",
        "span_m": "final['b_m']",
        "aspect_ratio": "cfg.AR",
        "root_chord_m": "final['c_root_m']",
        "tip_chord_m": "final['c_tip_m']",
        "MAC_m": "final['MAC_m']",
        "taper_ratio": "result.Wing_taper",
        "sweep_half_rad": "cfg.sweep_half",
        "sweep_half_deg": "derived from cfg.sweep_half",
        "sweep_tc_rad": "cfg.sweep_tc",
        "sweep_tc_deg": "derived from cfg.sweep_tc",
        "thickness_to_chord_root": "cfg.tc_root",
        "thickness_to_chord_mean": "cfg.tc_mean",
        "root_thickness_m": "derived from cfg.tc_root * final['c_root_m']",
        "LEMAC_m_from_nose": "cfg.LEMAC",
        "loading_N_m2": "cfg.Loading",
    },

    "Horizontal_tail": {
        "S_h_m2": "result.tail_rechecked.S_h",
        "span_h_m": "result.tail_rechecked.b_h",
        "MAC_h_m": "cfg.MAC_h",
        "tail_arm_l_h_m": "cfg.l_h",
        "aspect_ratio_h": "cfg.AR_h",
        "sweep_h_half_rad": "cfg.sweep_h_half",
        "sweep_h_half_deg": "derived from cfg.sweep_h_half",
        "sweep_h_tc_rad": "cfg.sweep_h_tc",
        "sweep_h_tc_deg": "derived from cfg.sweep_h_tc",
        "thickness_to_chord_h": "cfg.tc_h",
        "V_h": "result.tail_rechecked.V_h",
        "elevator_area_m2": "result.tail_rechecked.S_elevator",
        "Se_Sh": "result.tail_rechecked.Se_Sh",
        "sizing_driver": "result.tail_rechecked.S_h_driver",
    },

    "Vertical_tail": {
        "S_v_m2": "result.tail_rechecked.S_v",
        "span_v_m": "result.tail_rechecked.b_v",
        "MAC_v_m": "cfg.MAC_v",
        "tail_arm_l_v_m": "cfg.l_v",
        "aspect_ratio_v": "cfg.AR_v",
        "sweep_v_half_rad": "cfg.sweep_v_half",
        "sweep_v_half_deg": "derived from cfg.sweep_v_half",
        "sweep_v_tc_rad": "cfg.sweep_v_tc",
        "sweep_v_tc_deg": "derived from cfg.sweep_v_tc",
        "thickness_to_chord_v": "cfg.tc_v",
        "V_v": "result.tail_rechecked.V_v",
        "rudder_area_m2": "result.tail_rechecked.S_rudder",
        "Sr_Sv": "result.tail_rechecked.Sr_Sv",
        "sizing_driver": "result.tail_rechecked.S_v_driver",
    },

    "Fuselage": {
        "length_m": "final['l_f_m']",
        "diameter_equivalent_m": "final['d_f_m']",
        "width_m": "cfg.b_f",
        "height_m": "cfg.h_f",
        "wetted_area_m2": "final['S_wet_f_m2']",
        "tail_length_l_t_m": "cfg.l_t",
        "LEMAC_m_from_nose": "cfg.LEMAC",
    },

    "LH2_tank": {
        "fuel_mass_kg": "result.W_fuel",
        "effective_LH2_density_kg_m3": "cfg.rho_LH2_eff",
        "volume_m3": "derived from result.W_fuel / cfg.rho_LH2_eff",
        "radius_m": "final['r_tank_m']",
        "diameter_m": "final['d_tank_m']",
        "length_m": "final['L_tank_m']",
        "hump_back_input_flag": "cfg.hump_back",
        "extra_wetted_area_hump_m2": "final['S_wet_hump_m2']",
        "tank_weight_kg": "result.weight.W_h2_tank",
        "gravimetric_density": "result.weight.grav_density",
    },

    "Layout_and_CG_inputs": {
        "PaxWeight_kg": "cfg.PaxWeight",
        "Pax_count": "cfg.Pax_count",
        "Seats_abreast": "cfg.Seats_abreast",
        "FirstWindow_m_from_nose": "cfg.FirstWindow",
        "LastWindow_m_from_nose": "cfg.LastWindow",
        "OEW_cg_m_from_nose": "cfg.OEW_cg",
        "FUEL_cg_m_from_nose": "cfg.FUEL_cg",
        "AftCargo_cg_m_from_nose": "cfg.AftCargo_cg",
        "FwdCargo_cg_m_from_nose": "cfg.FwdCargo_cg",
        "Max_fwd_cargo_vol_m3": "cfg.Max_fwd_cargo_vol",
        "Max_aft_cargo_vol_m3": "cfg.Max_aft_cargo_vol",
    },

    "Aerodynamics": {
        "cruise_CL": "result.CL_cruise",
        "cruise_L_over_D": "result.L_over_D",
        "CD0": "result.drag.CD0",
        "CD_total": "result.drag.CD_total",
        "CD_wave": "result.drag.CD_wave",
        "oswald_e": "result.drag.e",
        "CL_cruise_Dmin": "result.drag.CL_cruise_Dmin",
        "CD_cruise_Dmin": "result.drag.CD_cruise_Dmin",
        "v_dmin_m_s": "result.drag.v_dmin",
    },

    "Power_and_mission": {
        "P_cruise_kW": "result.P_cruise_KW",
        "P_climb_kW": "result.P_climb_KW",
        "P_reserve_kW": "result.P_reserve_KW",
        "P_TO_kW": "result.P_TO_KW",
        "P_TO_OEI_kW": "result.P_TO_OEI_KW",
        "P_max_kW": "result.P_max_KW",
        "static_thrust_per_engine_kN": "derived from result.power.T_static_per_engine / 1000",
        "static_thrust_total_kN": "derived from result.power.T_static_total / 1000",
        "t_cruise_s": "result.mission.t_cruise",
        "t_climb_s": "result.mission.t_climb",
        "t_reserve_s": "result.mission.t_reserve",
        "mission_fuel_total_kg": "result.mission.m_LH2_total",
    },

    "Iteration_info": {
        "iterations": "result.iterations",
        "converged": "result.converged",
        "final_delta_kg": "final['delta_kg']",
        "bt_charging_ratio": "final['bt_ch_ratio']",
    },
}


def _deg(rad: float) -> float:
    return float(rad) * 180.0 / 3.141592653589793


def collect_final_geometry(cfg: Any, result: Any, *, include_sources: bool = False) -> dict[str, Any]:
    """
    Collect the final aircraft geometry after the Class II iteration.

    This function does not change any sizing calculations. It only extracts
    values from AircraftConfig, ClassIIResult, result.iteration_log[-1], and
    the final tail recheck.
    """
    if not result.iteration_log:
        raise ValueError("No iteration log found. Run run_class_ii() before exporting geometry.")

    final = result.iteration_log[-1]

    tank_volume = result.W_fuel / cfg.rho_LH2_eff if cfg.rho_LH2_eff > 0 else 0.0

    geometry: dict[str, Any] = {
        "Masses": {
            "MTOW_kg": result.MTOW,
            "MZFW_kg": result.MZFW,
            "OEW_kg": result.OEW,
            "empty_weight_without_fixed_kg": result.W_empty,
            "fuel_mass_kg": result.W_fuel,
            "payload_kg": result.W_payload,
            "fixed_equipment_kg": result.W_fixed,
            "propulsion_system_with_tank_kg": result.W_prop,
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
            "sweep_half_deg": _deg(cfg.sweep_half),
            "sweep_tc_rad": cfg.sweep_tc,
            "sweep_tc_deg": _deg(cfg.sweep_tc),
            "thickness_to_chord_root": cfg.tc_root,
            "thickness_to_chord_mean": cfg.tc_mean,
            "root_thickness_m": cfg.tc_root * final["c_root_m"],
            "LEMAC_m_from_nose": cfg.LEMAC,
            "loading_N_m2": cfg.Loading,
        },

        "Horizontal_tail": {
            "S_h_m2": result.tail_rechecked.S_h,
            "span_h_m": result.tail_rechecked.b_h,
            "MAC_h_m": cfg.MAC_h,
            "tail_arm_l_h_m": cfg.l_h,
            "aspect_ratio_h": cfg.AR_h,
            "sweep_h_half_rad": cfg.sweep_h_half,
            "sweep_h_half_deg": _deg(cfg.sweep_h_half),
            "sweep_h_tc_rad": cfg.sweep_h_tc,
            "sweep_h_tc_deg": _deg(cfg.sweep_h_tc),
            "thickness_to_chord_h": cfg.tc_h,
            "V_h": result.tail_rechecked.V_h,
            "elevator_area_m2": result.tail_rechecked.S_elevator,
            "Se_Sh": result.tail_rechecked.Se_Sh,
            "sizing_driver": result.tail_rechecked.S_h_driver,
        },

        "Vertical_tail": {
            "S_v_m2": result.tail_rechecked.S_v,
            "span_v_m": result.tail_rechecked.b_v,
            "MAC_v_m": cfg.MAC_v,
            "tail_arm_l_v_m": cfg.l_v,
            "aspect_ratio_v": cfg.AR_v,
            "sweep_v_half_rad": cfg.sweep_v_half,
            "sweep_v_half_deg": _deg(cfg.sweep_v_half),
            "sweep_v_tc_rad": cfg.sweep_v_tc,
            "sweep_v_tc_deg": _deg(cfg.sweep_v_tc),
            "thickness_to_chord_v": cfg.tc_v,
            "V_v": result.tail_rechecked.V_v,
            "rudder_area_m2": result.tail_rechecked.S_rudder,
            "Sr_Sv": result.tail_rechecked.Sr_Sv,
            "sizing_driver": result.tail_rechecked.S_v_driver,
        },

        "Fuselage": {
            "length_m": final["l_f_m"],
            "diameter_equivalent_m": final["d_f_m"],
            "width_m": cfg.b_f,
            "height_m": cfg.h_f,
            "wetted_area_m2": final["S_wet_f_m2"],
            "tail_length_l_t_m": cfg.l_t,
            "LEMAC_m_from_nose": cfg.LEMAC,
        },

        "LH2_tank": {
            "fuel_mass_kg": result.W_fuel,
            "effective_LH2_density_kg_m3": cfg.rho_LH2_eff,
            "volume_m3": tank_volume,
            "radius_m": final["r_tank_m"],
            "diameter_m": final["d_tank_m"],
            "length_m": final["L_tank_m"],
            "hump_back_input_flag": cfg.hump_back,
            "extra_wetted_area_hump_m2": final["S_wet_hump_m2"],
            "tank_weight_kg": result.weight.W_h2_tank,
            "gravimetric_density": result.weight.grav_density,
        },

        "Layout_and_CG_inputs": {
            "PaxWeight_kg": cfg.PaxWeight,
            "Pax_count": cfg.Pax_count,
            "Seats_abreast": cfg.Seats_abreast,
            "FirstWindow_m_from_nose": cfg.FirstWindow,
            "LastWindow_m_from_nose": cfg.LastWindow,
            "OEW_cg_m_from_nose": cfg.OEW_cg,
            "FUEL_cg_m_from_nose": cfg.FUEL_cg,
            "AftCargo_cg_m_from_nose": cfg.AftCargo_cg,
            "FwdCargo_cg_m_from_nose": cfg.FwdCargo_cg,
            "Max_fwd_cargo_vol_m3": cfg.Max_fwd_cargo_vol,
            "Max_aft_cargo_vol_m3": cfg.Max_aft_cargo_vol,
        },

        "Aerodynamics": {
            "cruise_CL": result.CL_cruise,
            "cruise_L_over_D": result.L_over_D,
            "CD0": result.drag.CD0,
            "CD_total": result.drag.CD_total,
            "CD_wave": result.drag.CD_wave,
            "oswald_e": result.drag.e,
            "CL_cruise_Dmin": result.drag.CL_cruise_Dmin,
            "CD_cruise_Dmin": result.drag.CD_cruise_Dmin,
            "v_dmin_m_s": result.drag.v_dmin,
        },

        "Power_and_mission": {
            "P_cruise_kW": result.P_cruise_KW,
            "P_climb_kW": result.P_climb_KW,
            "P_reserve_kW": result.P_reserve_KW,
            "P_TO_kW": result.P_TO_KW,
            "P_TO_OEI_kW": result.P_TO_OEI_KW,
            "P_max_kW": result.P_max_KW,
            "static_thrust_per_engine_kN": result.power.T_static_per_engine / 1000.0,
            "static_thrust_total_kN": result.power.T_static_total / 1000.0,
            "t_cruise_s": result.mission.t_cruise,
            "t_climb_s": result.mission.t_climb,
            "t_reserve_s": result.mission.t_reserve,
            "mission_fuel_total_kg": result.mission.m_LH2_total,
        },

        "Iteration_info": {
            "iterations": result.iterations,
            "converged": result.converged,
            "final_delta_kg": final["delta_kg"],
            "bt_charging_ratio": final["bt_ch_ratio"],
        },
    }

    if include_sources:
        return {
            "values": geometry,
            "sources": SOURCE_MAP,
        }

    return geometry


def print_final_geometry(cfg: Any, result: Any, *, show_sources: bool = True) -> None:
    """
    Print a readable geometry summary in the console.
    """
    geometry = collect_final_geometry(cfg, result, include_sources=False)

    print("\n========== FINAL AIRCRAFT GEOMETRY ==========")

    for group, values in geometry.items():
        print(f"\n[{group}]")
        for key, value in values.items():
            source = SOURCE_MAP.get(group, {}).get(key, "unknown")
            if isinstance(value, float):
                value_str = f"{value:.6g}"
            else:
                value_str = str(value)

            if show_sources:
                print(f"{key:35s}: {value_str:>14s}    source: {source}")
            else:
                print(f"{key:35s}: {value_str:>14s}")


def export_final_geometry(
    cfg: Any,
    result: Any,
    output_dir: str | Path = "outputs",
    *,
    include_sources: bool = True,
) -> dict[str, Path]:
    """
    Export final geometry to JSON and CSV.

    JSON keeps the grouped structure. CSV is easier to inspect in Excel.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_data = collect_final_geometry(cfg, result, include_sources=include_sources)

    json_path = output_dir / "final_aircraft_geometry.json"
    csv_path = output_dir / "final_aircraft_geometry.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=4)

    values = export_data["values"] if include_sources else export_data
    sources = export_data["sources"] if include_sources else SOURCE_MAP

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "parameter", "value", "source"])

        for group, group_values in values.items():
            for key, value in group_values.items():
                source = sources.get(group, {}).get(key, "unknown")
                writer.writerow([group, key, value, source])

    return {
        "json": json_path,
        "csv": csv_path,
    }