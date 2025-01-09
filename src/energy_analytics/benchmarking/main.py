import os
import pandas as pd
from loguru import logger

from settings import PROJECT_ROOT
from .clustering import run_clustering
from .assign import calculate_medioids_profile, assign_cluster
from .viz import (plot_load_profiles_user, plot_load_profiles_aggregate,
                  plot_cluster_percentage, plot_feature_distribution)
from ..aggregate import Aggregate


def run_benchmarking(aggregate: Aggregate):
    """
    Esegue la pipeline di benchmarking per un intero aggregato, formata dal clustering e dalla visualizzazione dei risultati.

    Args:
        aggregate (Aggregate): Oggetto Aggregate con la lista di edifici (List[Building]).

    Returns:
        None
    """

    logger.info(f"Iniziando la pipeline di benchmarking esterno per l'aggregato {aggregate.name}")

    run_clustering(aggregate)
    calculate_medioids_profile(aggregate)
    assign_cluster(aggregate)
    plot_load_profiles_aggregate(aggregate)
    plot_cluster_percentage(aggregate)
    plot_feature_distribution(aggregate)

    cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "benchmarking", f"cluster_{aggregate.name}_assigned.csv"))

    for building in aggregate.buildings:
        plot_load_profiles_user(building, cluster)
