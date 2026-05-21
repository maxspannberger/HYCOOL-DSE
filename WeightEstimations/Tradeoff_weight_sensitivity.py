import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from General.component_parameters import component_params as comp_params
from Aircraft_Config import AircraftConfig, default_q400_hycool
from mainClassII import run_class_ii
from Climate_Impact.Average_Temp_Response import get_results as get_climate_results

from rich import print

# =============================================================================
# Calculate the performance data for the configurations to be considered
# =============================================================================
def calculate_static_performance(
        cfg: AircraftConfig,
        comp_params: dict = comp_params,
        designs_to_consider: list = [1, 2, 3, 4]
) -> dict:
    print("[bold green]Harvesting aircraft configuration performance metrics (Once)...[/bold green]")
    
    performance = {}
    climate_results = get_climate_results(comp=comp_params)
    design_names = list(climate_results.keys())

    for config in designs_to_consider:
        design_name = design_names[config - 1]
        print(f"\n[yellow]Calculating Class II sizing data for {design_name}")
        
        # Run class II calculations
        class_II_results = run_class_ii(config=config, comp=comp_params, verbose=False, cfg=cfg)

        performance[design_name] = {
            "OEW": class_II_results.W_empty,
            "prop_frac": class_II_results.W_prop / class_II_results.W_empty,
            "eff": class_II_results.total_prop_efficiency,
            "atr_ratio": 1 - climate_results[design_name] / climate_results["Baseline"]
        }
        
    return performance


# =============================================================================
# Calculate the baseline weights for the trade-off
# =============================================================================
def assign_scores(sizing_outputs, noise_scores):
    thermal_score = TRL_score = mass_score = eff_score = climate_score = 0

    prop_frac = sizing_outputs["prop_frac"]
    eff = sizing_outputs["eff"]
    atr_ratio = sizing_outputs["atr_ratio"]

    # Mass scoring
    if prop_frac < 0.15: mass_score = 5
    elif prop_frac < 0.20: mass_score = 4
    elif prop_frac < 0.25: mass_score = 3
    elif prop_frac < 0.30: mass_score = 2
    else: mass_score = 1

    # Efficiency scoring
    if eff < 0.40: eff_score = 1
    elif eff < 0.45: eff_score = 2
    elif eff < 0.50: eff_score = 3
    elif eff < 0.55: eff_score = 4
    else: eff_score = 5

    # Climate scoring
    if atr_ratio <= 0.00: climate_score = 1
    elif atr_ratio <= 0.25: climate_score = 2
    elif atr_ratio <= 0.50: climate_score = 3
    elif atr_ratio <= 0.75: climate_score = 4
    else: climate_score = 5

    # Dot product calculation for final trade-off score
    overall_score = round(
        noise_scores[0] * thermal_score + \
        noise_scores[1] * TRL_score + \
        noise_scores[2] * mass_score + \
        noise_scores[3] * eff_score + \
        noise_scores[4] * climate_score, 3)
    
    return {
        "mass": mass_score,
        "thermal": thermal_score,
        "efficiency": eff_score,
        "climate": climate_score,
        "TRL": TRL_score,
        "overall": overall_score
    }

# =============================================================================
# Introduce Gaussian noice in the weights
# =============================================================================
def weight_sensitivity_analysis(performance, n_repeats=1000, ssd_fraction=0.7, plot=True):
    print(f"\n[bold blue]Run simulation {n_repeats} times with varying weights...[/bold blue]")
    
    initial_weights = np.array([0.25, 0.15, 0.25, 0.2, 0.15])
    tradeoff_history = {design: [] for design in performance.keys()}

    for run in range(n_repeats):
        # Generate random noice scores. The noice is clipped to the domain 
        # (0, 1) and is normalized to add to 1
        noise = np.random.normal(0.0, ssd_fraction * initial_weights)
        raw_noise_scores = initial_weights + noise
        clipped_noise_scores = np.clip(raw_noise_scores, 0, 1)
        noise_scores = clipped_noise_scores / np.sum(clipped_noise_scores)

        # Score all configurations using the weight of the current configuration
        for design_name in performance:
            scores = assign_scores(performance[design_name], noise_scores)
            tradeoff_history[design_name].append(scores["overall"])

    # Compute ssd and mean per configuration and print
    print("\n[yellow]Weight Sensitivity Summary Results:")
    for design_name, results in tradeoff_history.items():
        print(f"{design_name:25} -> Mean Score: {np.mean(results):.3f} | Standard Dev: {np.std(results):.3f}")
        
    if plot:
        combined_results = []
        labels = []
        for design_name, results in tradeoff_history.items():
            combined_results.append(results)
            labels.append(design_name)
            
        plot = plt.boxplot(combined_results, labels=labels, patch_artist=True)
        
        # Set distinct colours to each box
        for i, box in enumerate(plot['boxes']):
            box.set_facecolor(plt.cm.tab10(i))
            
        plt.legend(plot['boxes'], labels, title="Configurations")
        plt.show()
                
    return tradeoff_history

    



if __name__ == "__main__":
    cfg = default_q400_hycool()
    designs = [1, 2, 3, 4]

    # Run Class II and Climate functions to get oerformance data
    performance = calculate_static_performance(cfg=cfg, designs_to_consider=designs)
    
    # Perform weight sensitivity analysis
    weight_sensitivity_analysis(performance, n_repeats=5000, ssd_fraction=0.7)
    
    