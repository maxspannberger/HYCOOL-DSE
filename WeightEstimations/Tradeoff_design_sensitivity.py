import numpy as np
import sys
from pathlib import Path

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from General.component_parameters import component_params as comp_params
from General.component_parameters import PowerComponent, StorageComponent, PipingComponent, CableComponent, HeatExchangeComponent
from Aircraft_Config import AircraftConfig, default_q400_hycool
from mainClassII import run_class_ii
from Climate_Impact.Average_Temp_Response import get_results as get_climate_results
from TMS.mainTMS import design_phase_table, design_score_table

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns


def sensitivity_analysis(
        cfg:            AircraftConfig,
        n_repeats:      int = 1,
        comp_params:    dict = comp_params,
        sensitivity_config: str = "none",
        designs_to_consider:list = [1, 2, 3, 4]
) -> dict:
    
    sensitivity_results = {}
    print(f"Starting sensitivity analysis...")
    max_runs = 1 if sensitivity_config == "none" else n_repeats

    for run in range(1, max_runs + 1):
        comp = comp_params.copy()
        sensitivity_results[run] = {}

        if sensitivity_config == "all":
            for param in comp:
                match comp[param]:

                    case PowerComponent():
                        comp[param].power_density += np.random.normal(0.0, comp[param].power_density_std)
                        if comp[param].power_density <= 0.0:
                            comp[param].power_density = 0.00001
                        comp[param].efficiency += np.random.normal(0.0, comp[param].efficiency_std)
                        if comp[param].efficiency <= 0.0:
                            comp[param].efficiency = 0.00001
                        elif comp[param].efficiency >= 1.0:
                            comp[param].efficiency = 0.99999

                    case StorageComponent():
                        comp[param].energy_density += np.random.normal(0.0, comp[param].energy_density_std)
                        if comp[param].energy_density <= 0.0:
                            comp[param].energy_density = 0.00001
                        comp[param].power_density += np.random.normal(0.0, comp[param].power_density_std)
                        if comp[param].power_density <= 0.0:
                            comp[param].power_density = 0.00001
                        comp[param].efficiency += np.random.normal(0.0, comp[param].efficiency_std)
                        if comp[param].efficiency <= 0.0:
                            comp[param].efficiency = 0.00001
                        elif comp[param].efficiency > 1.0:
                            comp[param].efficiency = 0.99999
                            
                    case _:
                        pass

        print(f"Performing run {run} out of {max_runs}")
        
        climate_results = get_climate_results(comp=comp)
        design_names = list(climate_results.keys())
        for config in designs_to_consider:
            class_II_results = run_class_ii(config=config, comp=comp, verbose=False, cfg=cfg)

            # TMS already has built-in scores
            TMS_results = design_phase_table(config=config, comp=comp)
            TMS_score = design_score_table(TMS_results)["FinalThermalScore"].iloc[0]

            sensitivity_results[run][design_names[config-1]] = {
                "OEW": class_II_results.W_empty,
                "prop_frac": class_II_results.W_prop / class_II_results.W_empty,
                "eff": class_II_results.total_prop_efficiency,
                "atr_ratio": 1 - climate_results[design_names[config-1]] / climate_results["Baseline"],
                "TMS_score": TMS_score
            }

    print(f"Sensitivity analysis finished.\n")
    return sensitivity_results


def stats_calculator(sensitivity_results):
    criterion_result = {}
    criterion_stats = {}

    for run in sensitivity_results:
        for config in sensitivity_results[run]:
            if config not in criterion_result:
                criterion_result[config] = {}

            for criterion in sensitivity_results[run][config]:
                if criterion in criterion_result:
                    criterion_result[config][criterion].append(sensitivity_results[run][config][criterion])
                else:
                    criterion_result[config][criterion] = [sensitivity_results[run][config][criterion]]

    for config in criterion_result:
        criterion_stats[config] = {}
        for criterion in criterion_result[config]:
            criterion_stats[config][criterion] = {
                "mean": np.mean(np.array(criterion_result[config][criterion])),
                "std": np.std(np.array(criterion_result[config][criterion]))
            }

    return criterion_stats


def assign_scores(sizing_outputs):
    thermal_score = TRL_score = mass_score = eff_score = climate_score = 0

    OEW = sizing_outputs["OEW"]
    prop_frac = sizing_outputs["prop_frac"]
    eff = sizing_outputs["eff"]
    atr_ratio = sizing_outputs["atr_ratio"]
    thermal_score = round(sizing_outputs["TMS_score"], 2)

    # mass scoring
    # TODO: decide if we choose OEW or prop_frac
    if prop_frac < 0.15:
        mass_score = 5
    elif prop_frac < 0.20:
        mass_score = 4
    elif prop_frac < 0.25:
        mass_score = 3
    elif prop_frac < 0.30:
        mass_score = 2
    else:
        mass_score = 1

    # efficiency scoring
    if eff < 0.40:
        eff_score = 1
    elif eff < 0.45:
        eff_score = 2
    elif eff < 0.50:
        eff_score = 3
    elif eff < 0.5:
        eff_score = 5
    else:
        eff_score = 5

    # climate scoring
    if atr_ratio <= 0.00:
        climate_score = 1
    elif atr_ratio <= 0.25:
        climate_score = 2
    elif atr_ratio <= 0.50:
        climate_score = 3
    elif atr_ratio <= 0.75:
        climate_score = 4
    else:
        climate_score = 5

    overall_score = round(
        0.25 * thermal_score +\
        0.15 * TRL_score +\
        0.25 * mass_score +\
        0.20 * eff_score +\
        0.15 * climate_score, 2)

    return {
        "mass": mass_score,
        "thermal": thermal_score,
        "efficiency": eff_score,
        "climate": climate_score,
        "TRL": TRL_score,
        "overall": overall_score
    }


def numerical_tradeoff(single_variation_results):
    tradeoff_table = {}
    for config in single_variation_results:
        if config not in tradeoff_table:
            tradeoff_table[config] = {}

        scores = assign_scores(single_variation_results[config])
        for criterion in scores:
            tradeoff_table[config][criterion] = scores[criterion]

    return tradeoff_table


def tradeoff_sensitivity(sensitivity_results):
    for run in sensitivity_results:
        tradeoff_table = numerical_tradeoff(sensitivity_results[run])
        print(tradeoff_table)



if __name__ == "__main__":
    cfg = default_q400_hycool()
    n_repeats = 3
    designs_to_consider = [1, 2, 3, 4]

    sensitivity_results = sensitivity_analysis(cfg=cfg, n_repeats=n_repeats, sensitivity_config="all", designs_to_consider=designs_to_consider)
    print(sensitivity_results)
    
    results_stats = stats_calculator(sensitivity_results)
    print(results_stats)

    tradeoff_sensitivity(sensitivity_results)