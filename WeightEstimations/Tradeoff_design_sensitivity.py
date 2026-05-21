import numpy as np
import sys
from pathlib import Path
from statistics import mean, stdev
import matplotlib.pyplot as plt
import json

# Add parent directory to path so General module can be imported
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from General.component_parameters import component_params as comp_params
from General.component_parameters import PowerComponent, StorageComponent
from Aircraft_Config import AircraftConfig, default_q400_hycool
from mainClassII import run_class_ii
from Climate_Impact.Average_Temp_Response import get_results as get_climate_results
from TMS.mainTMS import design_phase_table, design_score_table, thermal_ratio_score

from rich import print
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import copy


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
        case 5:
            return max(TRL["GT_TRL_base"] + TRL["GT_hex_TRL_penalty"], TRL["BAT_TRL_base"])
        case _:
            raise ValueError("Invalid configuration")


def single_sensitivity_run(
        run: int,
        cfg,
        comp_params,
        sensitivity_config,
        designs_to_consider,
        TRL_base,
        TRL_std
    ):
    try:
        comp = copy.deepcopy(comp_params)
        results = {}

        if sensitivity_config == "all":
            for param in comp:
                match comp[param]:

                    case PowerComponent():
                        comp[param].power_density += comp[param].power_density_std * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)
                        comp[param].power_density = max(comp[param].power_density, 0.1)

                        comp[param].efficiency += comp[param].efficiency_std * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)
                        comp[param].efficiency = min(max(comp[param].efficiency, 0.1), 0.9999)

                    case StorageComponent():
                        comp[param].energy_density += comp[param].energy_density_std * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)
                        comp[param].energy_density = max(comp[param].energy_density, 0.1)
                        
                        comp[param].power_density += comp[param].power_density_std * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)
                        comp[param].power_density = max(comp[param].power_density, 0.1)

                        comp[param].efficiency += comp[param].efficiency_std * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0)
                        comp[param].efficiency = min(max(comp[param].efficiency, 0.1), 0.9999)
                            
                    case _:
                        pass

            TRL = {}
            for component in TRL_base:
                TRL[component] = max(0, TRL_base[component] + TRL_std[component] * np.clip(np.random.normal(0.0, 1.0), -1.0, 1.0))

        print(f"Performing run {run}")

        climate_results = get_climate_results(comp=comp)
        design_names = list(climate_results.keys())
        for config in designs_to_consider:
            class_II_results = run_class_ii(config=config, comp=comp, verbose=False, cfg=cfg)

            # TMS already has built-in scores
            TMS_results = design_phase_table(config=config, comp=comp, class_II_results=class_II_results)
            TMS_ratio = design_score_table(TMS_results)["FinalRatio"].iloc[0]

            TRL_year = TRL_per_design(TRL, config=config)

            results[design_names[config-1]] = {
                # "OEW": class_II_results.W_empty,
                "prop_frac": class_II_results.W_prop / class_II_results.W_empty,
                "eff": class_II_results.total_prop_efficiency,
                "atr_ratio": 1 - climate_results[design_names[config-1]] / climate_results["Baseline"],
                "TMS_ratio": TMS_ratio,
                "TRL_year": TRL_year
            }

        return run, results, False
        
    except ValueError:
        print(f"Run {run} failed.")
        return run, {}, True



def sensitivity_analysis(
        cfg:            AircraftConfig,
        n_repeats:      int = 1,
        comp_params:    dict = comp_params,
        sensitivity_config: str = "none",
        designs_to_consider:list = [1, 2, 3, 4, 5]
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
    # uncertainties eyeballed by Francisco
    TRL_std = {
        "GT_TRL_base": 1,
        "FC_TRL_base": 1,
        "BAT_TRL_base": 2,
        "GT_hex_TRL_penalty": 1,
        "FC_hex_TRL_penalty": 2,
        "S_duct_TRL_penalty": 0.2,
        "Hump_TRL_penalty": 3,
        "Belly_FC_TRL_penalty": 0.5,
    }

    sensitivity_results = {}
    n_skipped = 0
    
    n_cores = multiprocessing.cpu_count()
    n_workers = max(1, int(n_cores * 0.9))
    print(f"Using {n_workers}/{n_cores} CPU cores")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                single_sensitivity_run,
                run,
                cfg,
                comp_params,
                sensitivity_config,
                designs_to_consider,
                TRL_base,
                TRL_std
            )
            for run in range(1, max_runs + 1)
        ]

        for future in as_completed(futures):
            run, result, failed = future.result()
            if failed:
                n_skipped += 1
            else:
                sensitivity_results[run] = result

    print("Sensitivity analysis finished.\n")
    return sensitivity_results, n_skipped


def stats_calculator(sensitivity_results, n_repeats=1, prefix=""):
    criterion_result = {}
    criterion_stats = {}

    for run in sensitivity_results:
        for config in sensitivity_results[run]:
            if config not in criterion_result:
                criterion_result[config] = {}

            for criterion in sensitivity_results[run][config]:
                if criterion in criterion_result[config]:
                    criterion_result[config][criterion].append(sensitivity_results[run][config][criterion])
                else:
                    criterion_result[config][criterion] = [sensitivity_results[run][config][criterion]]

    for config in criterion_result:
        criterion_stats[config] = {}
        for criterion in criterion_result[config]:
            criterion_stats[config][criterion] = {
                "mean": mean(criterion_result[config][criterion]),
                "std": stdev(criterion_result[config][criterion])
            }

    with open(f"{prefix}_tradeoff_quantity_stats_{n_repeats}_runs.json", "w") as f:
        json.dump(criterion_stats, f, indent=4)

    return criterion_stats


def assign_scores(sizing_outputs):
    TMS_score = TRL_score = mass_score = eff_score = climate_score = 0

    # OEW = sizing_outputs["OEW"]
    prop_frac = sizing_outputs["prop_frac"]
    eff = sizing_outputs["eff"]
    atr_ratio = sizing_outputs["atr_ratio"]
    TMS_ratio = round(sizing_outputs["TMS_ratio"], 2)
    TRL_year = sizing_outputs["TRL_year"]

    # mass fraction scoring
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
    if eff < 0.30:
        eff_score = 1
    elif eff < 0.37:
        eff_score = 2
    elif eff < 0.44:
        eff_score = 3
    elif eff < 0.51:
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

    # thermal scoring
    TMS_score = thermal_ratio_score(TMS_ratio)

    # TRL scoring
    if TRL_year <= 2035:
        TRL_score = 5
    elif TRL_year <= 2038:
        TRL_score = 4
    elif TRL_year <= 2041:
        TRL_score = 3
    elif TRL_year <= 2044:
        TRL_score = 2
    else:
        TRL_score = 1

    overall_score = round(
        0.25 * TMS_score +\
        0.15 * TRL_score +\
        0.25 * mass_score +\
        0.20 * eff_score +\
        0.15 * climate_score, 2)

    return {
        "mass": mass_score,
        "thermal": TMS_score,
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


def get_score_list(tradeoff_table_history, n_repeats=1, prefix=""):
    design_scores = {}
    for table in tradeoff_table_history:
        for config in table:
            if config not in design_scores:
                design_scores[config] = {}
            for metric in table[config]:
                if metric not in design_scores[config]:
                    design_scores[config][metric] = []
                design_scores[config][metric].append(table[config][metric])

    with open(f"{prefix}_tradeoff_score_lists_{n_repeats}_runs.json", "w") as f:
        json.dump(design_scores, f, indent=4)
    
    return design_scores


def get_score_uncertainties(design_scores, n_repeats=1, prefix=""):
    results = {}
    for config in design_scores:
        if config not in results:
            results[config] = {}
        for metric in design_scores[config]:
            if metric not in results[config]:
                results[config][metric] = {}
            results[config][metric]["mean"] = round(mean(design_scores[config][metric]), 4)
            results[config][metric]["std"] = round(stdev(design_scores[config][metric]), 4)

    with open(f"{prefix}_tradeoff_scores_stats_{n_repeats}_runs.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


def plot_scores(design_scores, n_repeats=1, n_skipped=0, show=False, prefix=""):
    fig, ax = plt.subplots()
    ticks = list(design_scores.keys())
    values = [design_scores[tick]["overall"] for tick in ticks]
    ax.boxplot(values, tick_labels=ticks)

    ax.set_ylim((0, 5))
    if n_skipped > 0:
        ax.set_title(f"Tradeoff sensitivity analysis for {n_repeats} runs ({n_skipped} skipped)")
    else:
        ax.set_title(f"Tradeoff sensitivity analysis for {n_repeats} runs")
    ax.set_xlabel("Design")
    ax.set_ylabel("Score")

    plt.savefig(f"{prefix}_tradeoff_design_sensitivity_analysis_{n_repeats}_runs")
    
    if show:
        plt.show()


def perform_sensitivity_analysis(cfg, n_repeats=1, designs_to_consider=[1,2,3,4,5], show=False, prefix=""):
    sensitivity_results, n_skipped = sensitivity_analysis(cfg=cfg, n_repeats=n_repeats, sensitivity_config="all", designs_to_consider=designs_to_consider)
    
    results_stats_metrics = stats_calculator(sensitivity_results, n_repeats=n_repeats, prefix=prefix)
    if show:
        print(results_stats_metrics)  

    tradeoff_table_history = tradeoff_sensitivity(sensitivity_results)
    design_scores = get_score_list(tradeoff_table_history, n_repeats=n_repeats, prefix=prefix)

    results_stats_scores = get_score_uncertainties(design_scores, n_repeats=n_repeats, prefix=prefix)
    if show:
        print(results_stats_scores)

    plot_scores(design_scores, n_repeats=n_repeats, n_skipped=n_skipped, show=show, prefix=prefix)

    return results_stats_metrics, results_stats_scores


if __name__ == "__main__":
    cfg = default_q400_hycool()
    n_repeats = 10
    designs_to_consider = [1, 2, 3, 4, 5]

    results_stats_metrics, results_stats_scores = perform_sensitivity_analysis(cfg=cfg,
                                                                               n_repeats=n_repeats,
                                                                               designs_to_consider=designs_to_consider,
                                                                               show=True)

