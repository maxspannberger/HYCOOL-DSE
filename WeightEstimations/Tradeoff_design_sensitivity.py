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
        designs_to_consider
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

        elif sensitivity_config == "none":
            TRL = TRL_base.copy()
        else:
            raise ValueError("Invalid sensitivity configuration")

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
                "TMS_ratio": TMS_ratio,
                "eff": class_II_results.total_prop_efficiency,
                "atr_ratio": 1 - climate_results[design_names[config-1]] / climate_results["Baseline"],
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
                designs_to_consider
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

    with open(f"sensitivity_outputs/{n_repeats}_runs/_sensitivity_raw_{n_repeats}_runs.json", "w") as f:
        json.dump(sensitivity_results, f, indent=4)

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

    with open(f"sensitivity_outputs/{n_repeats}_runs/{prefix}_tradeoff_quantity_stats_{n_repeats}_runs.json", "w") as f:
        json.dump(criterion_stats, f, indent=4)

    return criterion_stats


def assign_scores(sizing_outputs, weights):
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
        eff_score = 4
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

    overall_score = 0.0
    for criterion in weights:
        match criterion:
            case "mass":
                overall_score += weights[criterion] * mass_score
            case "thermal":
                overall_score += weights[criterion] * TMS_score
            case "efficiency":
                overall_score += weights[criterion] * eff_score
            case "climate":
                overall_score += weights[criterion] * climate_score
            case "TRL":
                overall_score += weights[criterion] * TRL_score

    return {
        "mass": mass_score,
        "thermal": TMS_score,
        "efficiency": eff_score,
        "climate": climate_score,
        "TRL": TRL_score,
        "overall": overall_score
    }


def numerical_tradeoff(single_variation_results, weights):
    tradeoff_table = {}
    for config in single_variation_results:
        if config not in tradeoff_table:
            tradeoff_table[config] = {}

        scores = assign_scores(single_variation_results[config], weights)
        for criterion in scores:
            tradeoff_table[config][criterion] = scores[criterion]

    return tradeoff_table


def tradeoff_sensitivity(sensitivity_results, weights):
    tradeoff_table_history = []
    for run in sensitivity_results:
        tradeoff_table = numerical_tradeoff(sensitivity_results[run], weights)
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

    with open(f"sensitivity_outputs/{n_repeats}_runs/{prefix}_tradeoff_score_lists_{n_repeats}_runs.json", "w") as f:
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

    with open(f"sensitivity_outputs/{n_repeats}_runs/{prefix}_tradeoff_scores_stats_{n_repeats}_runs.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


def plot_scores(design_scores, n_repeats=1, n_skipped=0, show=False, prefix="", for_weights=False, legend=False):
    if for_weights:
        name = "weight"
    else:
        name = "tradeoff"

    fig, ax = plt.subplots()
    ticks = list(design_scores.keys())
    if "overall" in design_scores[ticks[0]]:
        values = [design_scores[tick]["overall"] for tick in ticks]
    else:
        values = [design_scores[tick] for tick in ticks]
    plot = ax.boxplot(values, tick_labels=ticks, patch_artist=True)

    ax.set_ylim((0, 5))
    if n_skipped > 0:
        ax.set_title(f"{prefix} {name} sensitivity analysis for {n_repeats} runs ({n_skipped} skipped)")
    else:
        ax.set_title(f"{prefix} {name} sensitivity analysis for {n_repeats} runs")
    ax.set_xlabel("Design")
    ax.set_ylabel("Score")

    for i, box in enumerate(plot['boxes']):
        box.set_facecolor(plt.cm.tab10(i))
        
    if legend:
        plt.legend(plot['boxes'], ticks)

    plt.savefig(f"sensitivity_outputs/{n_repeats}_runs/{prefix}_{name}_design_sensitivity_analysis_{n_repeats}_runs")
    
    if show:
        plt.show()


def perform_sensitivity_analysis(cfg, n_repeats=1, designs_to_consider=[1,2,3,4,5], weights=None, from_file=True, show=False, prefix="", legend=False):
    if weights is None:
        weights={
            "mass": 0.25,
            "thermal": 0.25,
            "efficiency": 0.20,
            "climate": 0.15,
            "TRL": 0.15,
        }

    if from_file is True:
        with open(f"sensitivity_outputs/{n_repeats}_runs/_sensitivity_raw_{n_repeats}_runs.json", "r") as f:
            sensitivity_results = json.load(f)
            n_skipped = 0
    else:
        sensitivity_results, n_skipped = sensitivity_analysis(cfg=cfg, n_repeats=n_repeats, sensitivity_config="all", designs_to_consider=designs_to_consider)
    
    results_stats_metrics = stats_calculator(sensitivity_results, n_repeats=n_repeats, prefix=prefix)
    if show:
        print(results_stats_metrics)  

    tradeoff_table_history = tradeoff_sensitivity(sensitivity_results, weights)
    design_scores = get_score_list(tradeoff_table_history, n_repeats=n_repeats, prefix=prefix)

    results_stats_scores = get_score_uncertainties(design_scores, n_repeats=n_repeats, prefix=prefix)
    if show:
        print(results_stats_scores)

    plot_scores(design_scores, n_repeats=n_repeats, n_skipped=n_skipped, show=show, prefix=prefix, legend=legend)

    return results_stats_metrics, results_stats_scores


def eliminate_weight_criterion(base_weights, criterion=None):
    if criterion in base_weights:
        weights = base_weights.copy()
        del weights[criterion]
        sum_weights = sum(weights.values())
        for kept_criterion in weights:
            weights[kept_criterion] /= sum_weights
    else:
        raise ValueError(f"Criterion '{criterion}' invalid.")
    
    return weights


def eliminate_criteria(cfg, n_repeats=1, designs_to_consider=[1,2,3,4,5], base_weights=None):
    if base_weights is None:
        base_weights={
            "mass": 0.25,
            "thermal": 0.25,
            "efficiency": 0.20,
            "climate": 0.15,
            "TRL": 0.15,
        }
    
    for criterion in base_weights:
        weights = eliminate_weight_criterion(base_weights, criterion=criterion)

        results_stats_metrics, results_stats_scores = perform_sensitivity_analysis(cfg=cfg,
                                                                            n_repeats=n_repeats,
                                                                            designs_to_consider=designs_to_consider,
                                                                            weights=weights,
                                                                            from_file=True,
                                                                            show=False,
                                                                            prefix=f"NO_{criterion}")


def weight_sensitivity_analysis(cfg, n_repeats=1, designs_to_consider=[1,2,3,4,5], base_weights=None, ssd_fraction=0.7, plot=True, prefix=""):
    if base_weights is None:
        base_weights={
            "mass": 0.25,
            "thermal": 0.25,
            "efficiency": 0.20,
            "climate": 0.15,
            "TRL": 0.15,
        }

    performance = single_sensitivity_run(0, cfg, comp_params=comp_params, sensitivity_config="none", designs_to_consider=designs_to_consider)[1]
    tradeoff_history = {design: [] for design in performance.keys()}
    
    print(f"\n[bold blue]Run simulation {n_repeats} times with varying weights...[/bold blue]")
    for run in range(n_repeats):
        # Generate random noice scores. The noice is clipped to the domain 
        # (0, 1) and is normalized to add to 1
        clipped_noise_weights = {}
        for metric in base_weights:
            noise = np.random.normal(0.0, ssd_fraction * base_weights[metric])
            raw_noise_weights = base_weights[metric] + noise
            clipped_noise_weights[metric] = np.clip(raw_noise_weights, 0.0, 1.0)
        
        weights = {}
        weights_sum_initial = sum(list(clipped_noise_weights.values()))
        for metric in base_weights:
            weights[metric] = clipped_noise_weights[metric] / weights_sum_initial

        # Score all configurations using the weight of the current run
        for design_name in performance:
            scores = assign_scores(performance[design_name], weights=weights)
            tradeoff_history[design_name].append(scores["overall"])

    # Compute ssd and mean per configuration and print
    results_dict = {}
    print("\n[yellow]Weight Sensitivity Summary Results:")
    for design_name, results in tradeoff_history.items():
        results_dict[design_name] = {
            "mean": round(np.mean(results), 4),
            "std": round(np.std(results), 4)
        }
        print(f"{design_name:25} -> Mean Score: {results_dict[design_name]["mean"]} | Standard Dev: {results_dict[design_name]["std"]}")

    with open(f"sensitivity_outputs/{n_repeats}_runs/{prefix}_tradeoff_weight_stats_{n_repeats}_runs.json", "w") as f:
        json.dump(results_dict, f, indent=4)
        
    plot_scores(tradeoff_history, n_repeats=n_repeats, n_skipped=0, show=plot, prefix=prefix, for_weights=True)

    return results_dict


def save_tradeoff(cfg, designs_to_consider=[1,2,3,4,5]):
    tradeoff_metrics = single_sensitivity_run(0, cfg=cfg, comp_params=comp_params, sensitivity_config="none", designs_to_consider=designs_to_consider)[1]
    with open(f"sensitivity_outputs/tradeoff_metrics.json", "w") as f:
        json.dump(tradeoff_metrics, f, indent=4)

    tradeoff_scores = {}
    for design in tradeoff_metrics:
        tradeoff_scores[design] = assign_scores(tradeoff_metrics[design], weights=weights)
    with open(f"sensitivity_outputs/tradeoff_scores.json", "w") as f:
        json.dump(tradeoff_scores, f, indent=4)


if __name__ == "__main__":
    cfg = default_q400_hycool()
    n_repeats = 1000
    designs_to_consider = [1, 2, 3, 4, 5]
    weights={
        "mass": 0.25,
        "thermal": 0.25,
        "efficiency": 0.20,
        "climate": 0.20,
        "TRL": 0.10,
    }
    noise = 0.5

    results_stats_metrics, results_stats_scores = perform_sensitivity_analysis(cfg=cfg,
                                                                               n_repeats=n_repeats,
                                                                               designs_to_consider=designs_to_consider,
                                                                               weights=weights,
                                                                               from_file=False,
                                                                               show=True,
                                                                               legend=True)
    
    eliminate_criteria(cfg=cfg, n_repeats=n_repeats, designs_to_consider=designs_to_consider, base_weights=weights)

    weight_sensitivity_analysis(cfg=cfg, n_repeats=n_repeats, designs_to_consider=designs_to_consider, base_weights=weights,
                                ssd_fraction=noise, plot=True)
    
    save_tradeoff(cfg, designs_to_consider=designs_to_consider)

