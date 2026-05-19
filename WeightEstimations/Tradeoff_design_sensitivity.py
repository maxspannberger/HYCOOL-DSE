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
                    case StorageComponent():
                        comp[param].energy_density += np.random.normal(0.0, comp[param].energy_density_std)
                    case PipingComponent():
                        comp[param].mass_per_length += np.random.normal(0.0, comp[param].mass_per_length_std)
                    case CableComponent():
                        comp[param].power_density += np.random.normal(0.0, comp[param].power_density_std)
                    case HeatExchangeComponent():
                        pass
                    case _:
                        pass

        print(f"Performing run {run} out of {max_runs}")
        
        for config in designs_to_consider:
            class_II_results = run_class_ii(config=config, comp=comp, verbose=False, cfg=cfg)
            sensitivity_results[run][config] = {
                "MTOW": class_II_results.MTOW,
                "eff": class_II_results.total_prop_efficiency
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


def assign_scores(sensitivity_results):
    tradeoff_table = {
        1: None,
        2: None,
        3: None,
        4: None
    }

    for design in tradeoff_table:
        runs = sensitivity_results[design-1]



if __name__ == "__main__":
    cfg = default_q400_hycool()
    n_repeats = 3
    designs_to_consider = [1, 2, 3, 4]

    sensitivity_results = sensitivity_analysis(cfg=cfg, n_repeats=n_repeats, sensitivity_config="all", designs_to_consider=designs_to_consider)
    print(sensitivity_results)
    
    results_stats = stats_calculator(sensitivity_results)
    print(results_stats)