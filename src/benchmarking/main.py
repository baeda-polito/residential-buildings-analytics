import json
from src.benchmarking.clustering import run_clustering
from src.benchmarking.assign import calculate_medioids, assign_cluster
from src.benchmarking.viz import (plot_load_profiles_user, plot_load_profiles_aggregate,
                                  plot_cluster_percentage)


def run_benchmarking(aggregate: str):
    """
    Esegue la pipeline di benchmarking per un intero aggregato, formata dal clustering e dalla visualizzazione dei risultati.
    :param aggregate: nome dell'aggregato ("anguillara" o "garda")
    """

    with open(f"../../data/metadata/{aggregate}.json", "r") as f:
        building_list = json.load(f)

    # run_clustering(aggregate)
    calculate_medioids(aggregate)
    assign_cluster(aggregate)
    plot_load_profiles_aggregate(aggregate)
    plot_cluster_percentage(aggregate)

    for building in building_list:
        plot_load_profiles_user(building["id"], aggregate)


if __name__ == "__main__":
    run_benchmarking("anguillara")
