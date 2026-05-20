import numpy as np
import sys
from pathlib import Path
from statistics import mean, stdev
import matplotlib.pyplot as plt

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from General.component_parameters import component_params as comp_params
from General.component_parameters import PowerComponent, StorageComponent
from Aircraft_Config import AircraftConfig, default_q400_hycool
from mainClassII import run_class_ii
from Climate_Impact.Average_Temp_Response import get_results as get_climate_results
from TMS.mainTMS import design_phase_table, design_score_table

from rich import print


def TRL_per_design(TRL, config):
    match config:
        case 1:
            return max(TRL["GT_TRL_base"] + TRL["GT_hex_TRL_penalty"] + TRL["S_duct_TRL_penalty"], TRL["BAT_TRL_base"]) + TRL["Hump_TRL_penalty"]
        case 2:
            return max(TRL["FC_TRL_base"] + TRL["FC_hex_TRL_penalty"], TRL["BAT_TRL_base"]) 
        case 3:
            return TRL["GT_TRL_base"] + TRL["GT_hex_TRL_penalty"]
        case 4:
            return max(TRL["GT_TRL_base"] + TRL["GT_hex_TRL_penalty"], TRL["FC_TRL_base"] + TRL["FC_hex_TRL_penalty"] + TRL["Belly_FC_TRL_penalty"])
        case _:
            raise ValueError("Invalid configuration")


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

    # TRL
    TRL_base = {
        "GT_TRL_base": 2035,
        "FC_TRL_base": 2035,
        "BAT_TRL_base": 2040,
        "GT_hex_TRL_penalty": 2,
        "FC_hex_TRL_penalty": 5,
        "S_duct_TRL_penalty": 1,
        "Hump_TRL_penalty": 5,
        "Belly_FC_TRL_penalty": 2,
    }
    # TODO: add documented uncertainties
    TRL_std = {
        "GT_TRL_base": 1,
        "FC_TRL_base": 2,
        "BAT_TRL_base": 2,
        "GT_hex_TRL_penalty": 0.5,
        "FC_hex_TRL_penalty": 1,
        "S_duct_TRL_penalty": 0.2,
        "Hump_TRL_penalty": 1,
        "Belly_FC_TRL_penalty": 0.5,
    }

    n_skipped = 0
    for run in range(1, max_runs + 1):
        comp = comp_params.copy()
        sensitivity_results[run] = {}

        if sensitivity_config == "all":
            for param in comp:
                match comp[param]:

                    case PowerComponent():
                        comp[param].power_density += comp[param].power_density_std * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)
                        if comp[param].power_density <= 0.1:
                            comp[param].power_density = 0.1
                        comp[param].efficiency += comp[param].efficiency_std * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)
                        if comp[param].efficiency <= 0.1:
                            comp[param].efficiency = 0.1
                        elif comp[param].efficiency >= 0.9999:
                            comp[param].efficiency = 0.9999

                    case StorageComponent():
                        comp[param].energy_density += comp[param].energy_density_std * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)
                        if comp[param].energy_density <= 0.1:
                            comp[param].energy_density = 0.1
                        comp[param].power_density += comp[param].power_density_std * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)
                        if comp[param].power_density <= 0.1:
                            comp[param].power_density = 0.1
                        comp[param].efficiency += comp[param].efficiency_std * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)
                        if comp[param].efficiency <= 0.1:
                            comp[param].efficiency = 0.1
                        elif comp[param].efficiency > 0.9999:
                            comp[param].efficiency = 0.9999
                            
                    case _:
                        pass

            TRL = {}
            for component in TRL_base:
                TRL[component] = max(0, TRL_base[component] + TRL_std[component]) * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)

        print(f"Performing run {run} out of {max_runs}")
        try:
            climate_results = get_climate_results(comp=comp)
            design_names = list(climate_results.keys())
            for config in designs_to_consider:
                class_II_results = run_class_ii(config=config, comp=comp, verbose=False, cfg=cfg)

                # TMS already has built-in scores
                TMS_results = design_phase_table(config=config, comp=comp, class_II_results=class_II_results)
                TMS_score = design_score_table(TMS_results)["FinalThermalScore"].iloc[0]

                TRL_year = TRL_per_design(TRL, config=config)

                sensitivity_results[run][design_names[config-1]] = {
                    "OEW": class_II_results.W_empty,
                    "prop_frac": class_II_results.W_prop / class_II_results.W_empty,
                    "eff": class_II_results.total_prop_efficiency,
                    "atr_ratio": 1 - climate_results[design_names[config-1]] / climate_results["Baseline"],
                    "TMS_score": TMS_score,
                    "TRL_year": TRL_year
                }
        except ValueError:
            n_skipped += 1
            print(f"Run {run} failed.")

    print(f"Sensitivity analysis finished.\n")
    return sensitivity_results, n_skipped


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
    TRL_year = sizing_outputs["TRL_year"]

    # thermal scoring
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

    # TRL scoring
    if TRL_year < 2035:
        TRL_score = 5
    elif TRL_year < 2038:
        TRL_score = 4
    elif TRL_year < 2041:
        TRL_score = 3
    elif TRL_year < 2044:
        TRL_score = 2
    else:
        TRL_score = 1

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
    tradeoff_table_history = []
    for run in sensitivity_results:
        tradeoff_table = numerical_tradeoff(sensitivity_results[run])
        tradeoff_table_history.append(tradeoff_table)
    return tradeoff_table_history


def get_score_list(tradeoff_table_history):
    design_scores = {}
    for table in tradeoff_table_history:
        for config in table:
            if config not in design_scores:
                design_scores[config] = []
            design_scores[config].append(table[config]["overall"])
    return design_scores


def get_score_uncertainties(design_scores):
    results = {}
    for config in design_scores:
        if config not in results:
            results[config] = {}
        results[config]["mean"] = round(mean(design_scores[config]), 4)
        results[config]["std"] = round(stdev(design_scores[config]), 4)
    return results


def plot_scores(design_scores, n_repeats=1, n_skipped=0, show=False):
    fig, ax = plt.subplots()
    ax.boxplot(design_scores.values(), tick_labels=design_scores.keys())

    ax.set_ylim((0, 5))
    if n_skipped > 0:
        ax.set_title(f"Tradeoff sensitivity analysis for {n_repeats} runs ({n_skipped} skipped)")
    else:
        ax.set_title(f"Tradeoff sensitivity analysis for {n_repeats} runs")
    ax.set_xlabel("Design")
    ax.set_ylabel("Score")

    plt.savefig(f"Tradeoff_design_sensitivity_analysis_{n_repeats}_runs")
    
    if show:
        plt.show()


if __name__ == "__main__":
    cfg = default_q400_hycool()
    n_repeats = 10
    designs_to_consider = [1, 2, 3, 4]

    sensitivity_results, n_skipped = sensitivity_analysis(cfg=cfg, n_repeats=n_repeats, sensitivity_config="all", designs_to_consider=designs_to_consider)
    # print(sensitivity_results)
    
    results_stats = stats_calculator(sensitivity_results)
    # print(results_stats)

    tradeoff_table_history = tradeoff_sensitivity(sensitivity_results)
    # print(tradeoff_table_history)

    design_scores = get_score_list(tradeoff_table_history)
    # print(design_scores)

    results = get_score_uncertainties(design_scores)
    print(results)

    plot_scores(design_scores, n_repeats=n_repeats, n_skipped=n_skipped, show=True)