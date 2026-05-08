import numpy as np
from matplotlib import pyplot as plt

table = {
    "GT-BAT": {
        "GT-BAT-01": {
            "weight": 3,
            "stblty": 4,
            "safety": 2,
            "therml": 1
        },
        "GT-BAT-02": {
            "weight": 4,
            "stblty": 1,
            "safety": 3,
            "therml": 1
        },
        "GT-BAT-03": {
            "weight": 4,
            "stblty": 2,
            "safety": 4,
            "therml": 4
        },
        "GT-BAT-04": {
            "weight": 5,
            "stblty": 3,
            "safety": 3,
            "therml": 5
        },
        "GT-BAT-05": {
            "weight": 2,
            "stblty": 4,
            "safety": 3,
            "therml": 2
        },
        "GT-BAT-06": {
            "weight": 3,
            "stblty": 4,
            "safety": 4,
            "therml": 2
        },
        "GT-BAT-07": {
            "weight": 3,
            "stblty": 2,
            "safety": 3,
            "therml": 5
        },
        "GT-BAT-08": {
            "weight": 3,
            "stblty": 3,
            "safety": 2,
            "therml": 5
        },
        "GT-BAT-09": {
            "weight": 1,
            "stblty": 5,
            "safety": 2,
            "therml": 3
        },
        "GT-BAT-10": {
            "weight": 2,
            "stblty": 5,
            "safety": 3,
            "therml": 3
        },
        "GT-BAT-11": {
            "weight": 1,
            "stblty": 3,
            "safety": 1,
            "therml": 5
        }
    },
    "FC-BAT": {
        "FC-BAT-01": {
            "weight": 3,
            "stblty": 2,
            "safety": 3,
            "therml": 3
        },
        "FC-BAT-02": {
            "weight": 5,
            "stblty": 2,
            "safety": 5,
            "therml": 5
        },
        "FC-BAT-03": {
            "weight": 5,
            "stblty": 3,
            "safety": 4,
            "therml": 3
        },
        "FC-BAT-04": {
            "weight": 3,
            "stblty": 2,
            "safety": 2,
            "therml": 2
        },
        "FC-BAT-05": {
            "weight": 4,
            "stblty": 2,
            "safety": 4,
            "therml": 4
        },
        "FC-BAT-06": {
            "weight": 4,
            "stblty": 3,
            "safety": 4,
            "therml": 5
        },
        "FC-BAT-07": {
            "weight": 4,
            "stblty": 3,
            "safety": 3,
            "therml": 3
        },
        "FC-BAT-08": {
            "weight": 2,
            "stblty": 4,
            "safety": 4,
            "therml": 2
        },
        "FC-BAT-09": {
            "weight": 2,
            "stblty": 4,
            "safety": 3,
            "therml": 1
        },
        "FC-BAT-10": {
            "weight": 1,
            "stblty": 5,
            "safety": 1,
            "therml": 2
        },

    },
    "GT-GT": {
        "GT-GT-01": {
            "weight": 4,
            "stblty": 1,
            "safety": 3,
            "therml": 2
        },
        "GT-GT-02": {
            "weight": 5,
            "stblty": 3,
            "safety": 5,
            "therml": 3
        },
        "GT-GT-03": {
            "weight": 3,
            "stblty": 5,
            "safety": 4,
            "therml": 1
        },
        "GT-GT-04": {
            "weight": 4,
            "stblty": 3,
            "safety": 4,
            "therml": 3
        },
        "GT-GT-05": {
            "weight": 2,
            "stblty": 5,
            "safety": 2,
            "therml": 1
        },
        "GT-GT-06": {
            "weight": 1,
            "stblty": 2,
            "safety": 1,
            "therml": 4
        }
    },
    "GT-FC": {
        "GT-FC-01": {
            "weight": 3,
            "stblty": 1,
            "safety": 2,
            "therml": 4
        },
        "GT-FC-02": {
            "weight": 4,
            "stblty": 1,
            "safety": 3,
            "therml": 4
        },
        "GT-FC-03": {
            "weight": 4,
            "stblty": 3,
            "safety": 3,
            "therml": 4
        },
        "GT-FC-04": {
            "weight": 5,
            "stblty": 3,
            "safety": 4,
            "therml": 5
        },
        "GT-FC-05": {
            "weight": 2,
            "stblty": 4,
            "safety": 3,
            "therml": 3
        },
        "GT-FC-06": {
            "weight": 3,
            "stblty": 4,
            "safety": 4,
            "therml": 3
        },
        "GT-FC-07": {
            "weight": 3,
            "stblty": 3,
            "safety": 2,
            "therml": 4
        },
        "GT-FC-08": {
            "weight": 3,
            "stblty": 3,
            "safety": 3,
            "therml": 5
        },
        "GT-FC-09": {
            "weight": 1,
            "stblty": 5,
            "safety": 1,
            "therml": 2
        },
        "GT-FC-10": {
            "weight": 2,
            "stblty": 5,
            "safety": 2,
            "therml": 1
        },
        "GT-FC-11": {
            "weight": 1,
            "stblty": 2,
            "safety": 1,
            "therml": 4
        },
        "GT-FC-12": {
            "weight": 1,
            "stblty": 5,
            "safety": 1,
            "therml": 3
        }
    }
}


def tradeoff(weights, table=table):
    results = {}
    for design in table:
        results[design] = {}
        for config in table[design]:
            results[design][config] = 0.0
            for criterion in weights:
                results[design][config] += table[design][config][criterion] * weights[criterion]
            results[design][config] = round(results[design][config], 3)
    return results


def generate_adjusted_weights(weights, noise=0.0, n_runs=1):
    all_new_weights = []

    for _ in range(n_runs):
        new_weights = {}
        for criterion in weights:
            new_weights[criterion] = weights[criterion] * max(0.0, 1+noise*np.random.randn())

        weights_magnitude = 0.0
        for criterion in new_weights:
            weights_magnitude += new_weights[criterion]

        for criterion in new_weights:
            new_weights[criterion] = round(new_weights[criterion]/weights_magnitude, 3)
        all_new_weights.append(new_weights.copy())

    return all_new_weights


def get_winner(weights, n_winners=1):
    results = tradeoff(weights, table=table)
    winner = {}
    for place in range(1, 1+n_winners):
        for design in table:
            if design not in winner:
                winner[design] = {}
            winner[design][place] = max(results[design], key=results[design].get)
            del results[design][winner[design][place]] # delete the best so next time we can choose the second best and so on
    return winner


def get_winner_distribution(all_new_weights, n_winners=1):
    all_winners = {}
    for weights in all_new_weights:
        winners = get_winner(weights, n_winners=n_winners)
        for design in winners:
            for place in winners[design]:
                if design not in all_winners:
                    all_winners[design] = {}
                if winners[design][place] in all_winners[design]:
                    all_winners[design][winners[design][place]] += 1
                else:
                    all_winners[design][winners[design][place]] = 1
    return all_winners


def plot_winner_distribution(all_winners, n_winners=1, n_runs=1):
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 7.2))
    fig.subplots_adjust(hspace=1.0, bottom=0.2)
    axs = axs.reshape(-1)

    for ax, design in zip(axs, list(all_winners.keys())):
        design_scores = dict(sorted(all_winners[design].items(), key=lambda item: item[0]))
        ax.bar(design_scores.keys(), np.array(list(design_scores.values()))/n_runs)

        ax.set_title(design)
        ax.set_xlabel("Configuration")
        ax.set_ylabel(f"Occurance fraction in top {n_winners}")
        ax.set_ylim(0, 1)
        ax.tick_params("x", rotation=60)
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        # ax.grid()

    plt.savefig(f"Tradeoff_config_sensitivity_top_{n_winners}.png")
    plt.show()

    



if __name__ == "__main__":
    weights = {
        "weight": 0.3,
        "stblty": 0.3,
        "safety": 0.3,
        "therml": 0.1
    }

    noise = 0.7
    n_runs = 1000
    n_winners = 3

    all_new_weights = generate_adjusted_weights(weights, noise, n_runs)
    all_winners = get_winner_distribution(all_new_weights, n_winners=n_winners)
    plot_winner_distribution(all_winners, n_winners=n_winners, n_runs=n_runs)