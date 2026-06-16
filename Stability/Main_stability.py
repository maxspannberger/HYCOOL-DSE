from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import dataclass
from pprint import pprint
import copy
from dataclasses import replace

import numpy as np
from rich.console import Console
from rich.table import Table

# Allow running from inside Stability or from the project root
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from WeightEstimations.Aircraft_Config import default_q400_hycool
from WeightEstimations.mainClassII import run_class_ii
from General.component_parameters import component_params as comp_params

from Cg_Calculations import CgCalculationInput, CgCalculator
from ClassII_Loading_Diagram import LoadingDiagramInput, LoadingDiagramEstimator
from ClassII_Scissor_Plot import ScissorPlotInput, ScissorPlotEstimator

# Code required for verification
# =============================================================================
VALIDATION = False
PARAM  = 'W_fuel'

def adjust_param(obj, param_name=PARAM, factor=1e-10):
    """
    Manually adjusts a parameter by factor, handling nesting.
    """
    # Define which sub-object contains which parameter
    # Add new mappings here as you add more parameters
    mapping = {
        # --- Root Level ---
        'MTOW': ('root', 'MTOW'),
        'MZFW': ('root', 'MZFW'),
        'W_empty': ('root', 'W_empty'),
        'OEW': ('root', 'OEW'),
        'W_fuel': ('root', 'W_fuel'),
        'W_payload': ('root', 'W_payload'),
        'W_fixed': ('root', 'W_fixed'),
        'L_over_D': ('root', 'L_over_D'),
        
        # --- Tail Sizing ---
        'S_h': ('tail', 'S_h'),
        'S_v': ('tail', 'S_v'),
        'V_h': ('tail', 'V_h'),
        'S_elevator': ('tail', 'S_elevator'),
        'S_rudder': ('tail', 'S_rudder'),
        
        # --- Drag Breakdown ---
        'CD0_wing': ('drag', 'CD0_wing'),
        'CD0_total': ('drag', 'CD0_total'),
        'e': ('drag', 'e'),
        
        # --- Weight Breakdown ---
        'W_wing_accurate': ('weight', 'W_wing_accurate'),
        'W_h2_tank': ('weight', 'W_h2_tank'),
        'P_TO_KW': ('weight', 'P_TO_KW'),
        
        # --- Mission ---
        't_cruise': ('mission', 't_cruise'),
        'm_LH2_cruise': ('mission', 'm_LH2_cruise'),
        
        # --- Power ---
        'P_TO_total': ('power', 'P_TO_total'),
        'P_takeoff': ('power', 'P_takeoff'),
        
        # --- Aero Parameters (Special: Dictionary) ---
        'CD_Dmin': ('aeroparameters', 'CD_Dmin'),
        'CL_opt': ('aeroparameters', 'CL_opt')
    }

    if param_name not in mapping:
        raise ValueError(f"Parameter '{param_name}' not in ClassIIresults")

    group, field = mapping[param_name]

    if group == 'root':
        new_val = getattr(obj, param_name) * factor
        return replace(obj, **{param_name: new_val})
    else:
        nested_obj = getattr(obj, group)
        new_val = getattr(nested_obj, field) * factor
        updated_nested = replace(nested_obj, **{field: new_val})
        return replace(obj, **{group: updated_nested})

# =============================================================================


@dataclass
class FullStabilityRunOutput:
    cfg: object
    result: object
    cg_breakdown: object
    loading_breakdown: object
    scissor_breakdown: object

def print_master_summary(
    cg_breakdown,
    loading_breakdown,
    scissor_breakdown,
    cfg,
    l_nlg,
    nlg_load_frn,
) -> None:
    table = Table(title="Master Stability Iteration Summary")
    table.add_column("Quantity")
    table.add_column("Value", justify="right")
    table.add_column("Unit / note")

    table.add_row("OEW CG", f"{cg_breakdown.x_cg_OEW:.3f}", "m from nose")
    table.add_row("LEMAC", f"{cg_breakdown.X_LEMAC_new:.3f}", "m from nose")
    table.add_row("lfn", f"{cg_breakdown.lfn:.3f}", "m from nose")
    table.add_row("lfn", f"{cg_breakdown.lfn:.3f}", "m from nose")
    table.add_row("l_h", f"{cg_breakdown.l_h:.3f}", "m")
    table.add_row("l_nlg", f"{l_nlg:.3f}", "m")
    table.add_row("nlg_load_frn", f"{nlg_load_frn:.3f}", "load fraction")

    table.add_section()

    table.add_row(
        "xcg lower without margin",
        f"{loading_breakdown.xcg_lower:.3f}",
        "MAC fraction",
    )
    table.add_row(
        "xcg upper without margin",
        f"{loading_breakdown.xcg_upper:.3f}",
        "MAC fraction",
    )
    table.add_row(
        "xcg lower with margin",
        f"{loading_breakdown.xcg_lower_margin:.3f}",
        "MAC fraction",
    )
    table.add_row(
        "xcg upper with margin",
        f"{loading_breakdown.xcg_upper_margin:.3f}",
        "MAC fraction",
    )

    table.add_section()

    table.add_row(
        "Actual Sh/S",
        f"{scissor_breakdown.actual_ShS:.4f}",
        "from current tail size",
    )

    console = Console()
    console.print(table)


def run_full_stability_sequence(
    *,
    oew_target_rel_guess: float,
    config: int = 3,
    use_loading_margin: bool = True,
    #update_lfn: bool = True,
    show_plots: bool = True,
) -> FullStabilityRunOutput:
    """
    Runs the full chain:

    1. Set OEW/MAC fraction guess in cfg
    2. Run Class II
    3. Run CG calculation
    4. Update cfg with OEW CG, LEMAC, l_h, lfn and fuel CG
    5. Run loading diagram
    6. Update cfg with xcg_lower and xcg_upper
    7. Run scissor plot

    Tail size is still updated manually after reading the scissor plot.
    """

    cfg = default_q400_hycool()
    # ------------------------------------------------------------
    # Step 1: input OEW/MAC fraction guess
    # ------------------------------------------------------------
    cfg.OEW_target_rel = float(oew_target_rel_guess)

    # ------------------------------------------------------------
    # Step 2: run Class II once with this cfg
    # ------------------------------------------------------------
    result = run_class_ii(
        cfg,
        comp=comp_params,
        tol=1.0,
        max_iter=100,
        verbose=False,
        config=config,
    )
    if VALIDATION:
        result = adjust_param(result)
        
    pprint(result)
    
    
    

    # ------------------------------------------------------------
    # Step 3: run CG calculations
    # ------------------------------------------------------------
    cg_input = CgCalculationInput.from_class_ii(cfg, result)
    cg_estimator = CgCalculator(cg_input)
    cg_breakdown = cg_estimator.compute()

    # ------------------------------------------------------------
    # Step 4: update cfg with CG outputs
    # ------------------------------------------------------------
    cfg.LEMAC = float(cg_breakdown.X_LEMAC_new)
    cfg.l_h = float(cg_breakdown.l_h)

    # Global CG locations in metres from nose
    cfg.OEW_cg = float(cg_breakdown.x_cg_OEW)
    cfg.FUEL_cg = float(cg_breakdown.x_cg_tank)

    cfg.AftCargo_cg = float(cg_breakdown.x_cg_cargo_aft)
    cfg.FwdCargo_cg = float(cg_breakdown.x_cg_cargo_fwd)

    cfg.lfn = float(cg_breakdown.lfn)

    # ------------------------------------------------------------
    # Step 5: run loading diagram
    # ------------------------------------------------------------
    loading_input = LoadingDiagramInput.from_class_ii(cfg, result)
    loading_estimator = LoadingDiagramEstimator(loading_input)
    loading_breakdown = loading_estimator.compute()

    loading_estimator.plot(
        loading_breakdown,
        show=show_plots,
        save_path="loading_diagram_master.png",
    )

    # ------------------------------------------------------------
    # Step 6: update cfg with CG limits
    # ------------------------------------------------------------
    if use_loading_margin:
        cfg.xcg_lower = float(loading_breakdown.xcg_lower_margin)
        cfg.xcg_upper = float(loading_breakdown.xcg_upper_margin)
    else:
        cfg.xcg_lower = float(loading_breakdown.xcg_lower)
        cfg.xcg_upper = float(loading_breakdown.xcg_upper)

    # ------------------------------------------------------------
    # Step 7: run scissor plot
    # ------------------------------------------------------------
    aero_dict = result.aeroparameters

    scissor_input = ScissorPlotInput.from_class_ii(
        cfg,
        result,
        aero_dict=aero_dict,
    )

    scissor_estimator = ScissorPlotEstimator(scissor_input)
    scissor_breakdown = scissor_estimator.compute()

    scissor_estimator.print_debug(scissor_breakdown)
    scissor_estimator.plot(
        scissor_breakdown,
        show=show_plots,
        save_path="scissor_plot_master.png",
    )

    beta = cfg.beta
    beta_rad = np.pi / 180 * beta
    MAC = cfg.MAC
    z_cg = cfg.z_cg      #m = estimate for height of aricraft vertical cg
    x_cg_lg_main_frn = (z_cg*np.tan(beta_rad) + cfg.xcg_upper*MAC)/MAC
    x_cg_lg_main = x_cg_lg_main_frn*MAC + cfg.LEMAC
    l_cg_aft = cfg.xcg_upper*MAC + cfg.LEMAC
    l_cg_fwd = cfg.xcg_lower*MAC + cfg.LEMAC 

    l_nlg = ((1-0.08)*x_cg_lg_main-l_cg_aft)/(-0.08)

    nlg_load_frn = (x_cg_lg_main-l_cg_fwd)/(x_cg_lg_main-l_nlg)

    print_master_summary(
        cg_breakdown=cg_breakdown,
        loading_breakdown=loading_breakdown,
        scissor_breakdown=scissor_breakdown,
        cfg=cfg,
        l_nlg = l_nlg,
        nlg_load_frn = nlg_load_frn
    )

    return FullStabilityRunOutput(
        cfg=cfg,
        result=result,
        cg_breakdown=cg_breakdown,
        loading_breakdown=loading_breakdown,
        scissor_breakdown=scissor_breakdown,
    )


if __name__ == "__main__":
    run_full_stability_sequence(
        oew_target_rel_guess=0.50,   # your OEW/MAC fraction guess
        config=3,
        use_loading_margin=True,
        #update_lfn=True,
        show_plots=True,
    )