import matplotlib.pyplot as plt
from loguru import logger
import pandas as pd
import os

from settings import PROJECT_ROOT
from .viz import plot_daily_kpi, plot_kpi_distribution, plot_radar_user, plot_boxplot_kpi_aggregate
from ..aggregate import Aggregate
from .calculate import (calculate_kpi_load, calculate_kpi_flexibility, calculate_kpi_renewable,
                        calculate_performance_score_user, calculate_kpi_aggregate)


def run_kpi(aggregate: Aggregate, cluster: pd.DataFrame):
    """
    Esegue la pipeline di calcolo degli indicatori KPI per un intero aggregato. Salva i risultati nei folder figures/kpi and results/kpi.

    Args:
        aggregate (Aggregate): l'oggetto aggregato, composto da una lista di edifici.
        cluster (pd.DataFrame): il dataframe con i cluster assegnati per ogni edificio.
    Returns:
        None
    """

    logger.info(f"Iniziando la pipeline del calcolo dei KPI per l'aggregato {aggregate.name}")

    for building in aggregate.buildings:
        kpi_daily_load = calculate_kpi_load(building, cluster)
        fig_trend_load = plot_daily_kpi(kpi_daily_load["daily_kpi"].copy())
        plt.suptitle(f"Trend KPI giornalieri sul carico per l'edificio {building.building_info['name']}", fontsize=20,
                     fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.subplots_adjust(right=1.1)
        plt.savefig(os.path.join(PROJECT_ROOT, f"figures/kpi/trend_energy_{building.building_info['name']}.png"))

        fig_distribution_load = plot_kpi_distribution(kpi_daily_load["daily_kpi"])
        plt.suptitle(f"Distribuzione dei KPI giornalieri sul carico per l'edificio {building.building_info['name']}",
                     fontsize=20,
                     fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.savefig(os.path.join(PROJECT_ROOT, f"figures/kpi/distribution_energy_{building.building_info['name']}.png"))

        kpi_daily_flexibility = calculate_kpi_flexibility(building)
        fig_trend_flexibility = plot_daily_kpi(kpi_daily_flexibility["daily_kpi"].copy())
        plt.suptitle(f"Trend KPI giornalieri sulla flessibilità per l'edificio {building.building_info['name']}",
                     fontsize=20,
                     fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.subplots_adjust(right=1.1)
        plt.savefig(os.path.join(PROJECT_ROOT, f"figures/kpi/trend_flexibility_{building.building_info['name']}.png"))

        fig_distribution_flexibility = plot_kpi_distribution(kpi_daily_flexibility["daily_kpi"])
        plt.suptitle(
            f"Distribuzione dei KPI giornalieri sulla flessibilità per l'edificio {building.building_info['name']}",
            fontsize=20, fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.savefig(os.path.join(PROJECT_ROOT, f"figures/kpi/distribution_flexibility_{building.building_info['name']}.png"))

        if building.building_info["user_type"] != "consumer":
            kpi_daily_renewable = calculate_kpi_renewable(building)
            fig_trend_renewable = plot_daily_kpi(kpi_daily_renewable["daily_kpi"].copy())
            plt.suptitle(f"Trend KPI giornalieri sulle rinnovabili per l'edificio {building.building_info['name']}",
                         fontsize=20, fontweight="bold")
            plt.tight_layout(rect=(0, 0.03, 1, 0.99))
            plt.subplots_adjust(right=1.1)
            plt.savefig(os.path.join(PROJECT_ROOT, f"figures/kpi/trend_renewable_{building.building_info['name']}.png"))

            fig_distribution_renewable = plot_kpi_distribution(kpi_daily_renewable["daily_kpi"])
            plt.suptitle(
                f"Distribuzione dei KPI giornalieri sulle rinnovabili per l'edificio {building.building_info['name']}",
                fontsize=20, fontweight="bold")
            plt.tight_layout(rect=(0, 0.03, 1, 0.99))
            plt.subplots_adjust(right=0.98)
            plt.savefig(os.path.join(PROJECT_ROOT, f"figures/kpi/distribution_renewable_{building.building_info['name']}.png"))

        df_score_user = calculate_performance_score_user(building, cluster)
        fig_radar_user = plot_radar_user(df_score_user)
        plt.suptitle(f"Radar plot dei Performance Score per l'utente {building.building_info['name']}", fontsize=20,
                     fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.savefig(os.path.join(PROJECT_ROOT, f"figures/kpi/radar_user_{building.building_info['name']}.png"))

    df_kpi_load_aggregate, df_kpi_flexibility_aggregate, df_kpi_renewable_aggregate = calculate_kpi_aggregate(aggregate, cluster)

    df_kpi_load_aggregate.to_csv(os.path.join(PROJECT_ROOT, f"results/kpi/kpi_load_{aggregate.name}.csv"), index=False)
    df_kpi_flexibility_aggregate.to_csv(os.path.join(PROJECT_ROOT, f"results/kpi/kpi_flexibility_{aggregate.name}.csv"), index=False)
    df_kpi_renewable_aggregate.to_csv(os.path.join(PROJECT_ROOT, f"results/kpi/kpi_renewable_{aggregate.name}.csv"), index=False)

    fig_boxplot_kpi_load_aggregate = plot_boxplot_kpi_aggregate(df_kpi_load_aggregate)
    plt.suptitle(f"Boxplot dei KPI sul carico per l'aggregato {aggregate.name}", fontsize=20, fontweight="bold")
    plt.tight_layout(rect=(0, 0.03, 1, 0.99))
    plt.savefig(os.path.join(PROJECT_ROOT, f"figures/kpi/boxplot_load_{aggregate.name}.png"))

    fig_boxplot_kpi_flexibility_aggregate = plot_boxplot_kpi_aggregate(df_kpi_flexibility_aggregate)
    plt.suptitle(f"Boxplot dei KPI sulla flessibilità per l'aggregato {aggregate.name}", fontsize=20, fontweight="bold")
    plt.tight_layout(rect=(0, 0.03, 1, 0.99))
    plt.savefig(os.path.join(PROJECT_ROOT, f"figures/kpi/boxplot_flexibility_{aggregate.name}.png"))

    fig_boxplot_renewable_aggregate = plot_boxplot_kpi_aggregate(df_kpi_renewable_aggregate)
    plt.suptitle(f"Boxplot dei KPI sulle rinnovabili per l'aggregato {aggregate.name}", fontsize=20, fontweight="bold")
    plt.tight_layout(rect=(0, 0.03, 1, 0.99))
    plt.savefig(os.path.join(PROJECT_ROOT, f"figures/kpi/boxplot_renewable_{aggregate.name}.png"))