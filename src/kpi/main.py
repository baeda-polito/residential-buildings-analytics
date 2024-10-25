from src.kpi.viz import plot_daily_kpi, plot_kpi_distribution, plot_radar_user
from src.building import load_anguillara, load_garda
from src.kpi.calculate import (calculate_kpi_load, calculate_kpi_flexibility, calculate_kpi_renewable,
                               calculate_performance_score_user)
import matplotlib.pyplot as plt


def run_kpi(aggregate: str):
    """
    Esegue la pipeline di calcolo degli indicatori KPI per un intero aggregato.
    :param aggregate: il nome dell'aggregato ("anguillara" o "garda")
    """

    building_list = []
    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()

    for building in building_list:
        kpi_daily_load = calculate_kpi_load(building.building_info["id"], aggregate)
        fig_trend_load = plot_daily_kpi(kpi_daily_load["daily_kpi"].copy())
        plt.suptitle(f"Trend KPI giornalieri sul carico per l'edificio {building.building_info['name']}", fontsize=20,
                     fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.subplots_adjust(right=1.1)
        plt.savefig(f"../../figures/kpi/trend_energy_{building.building_info['name']}.png")

        fig_distribution_load = plot_kpi_distribution(kpi_daily_load["daily_kpi"])
        plt.suptitle(f"Distribuzione dei KPI giornalieri sul carico per l'edificio {building.building_info['name']}", fontsize=20,
                        fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.savefig(f"../../figures/kpi/distribution_energy_{building.building_info['name']}.png")

        kpi_daily_flexibility = calculate_kpi_flexibility(building.building_info["id"], aggregate)
        fig_trend_flexibility = plot_daily_kpi(kpi_daily_flexibility["daily_kpi"].copy())
        plt.suptitle(f"Trend KPI giornalieri sulla flessibilità per l'edificio {building.building_info['name']}", fontsize=20,
                     fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.subplots_adjust(right=1.1)
        plt.savefig(f"../../figures/kpi/trend_flexibility_{building.building_info['name']}.png")

        fig_distribution_flexibility = plot_kpi_distribution(kpi_daily_flexibility["daily_kpi"])
        plt.suptitle(f"Distribuzione dei KPI giornalieri sulla flessibilità per l'edificio {building.building_info['name']}",
                     fontsize=20, fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.savefig(f"../../figures/kpi/distribution_flexibility_{building.building_info['name']}.png")

        if building.building_info["user_type"] != "consumer":
            kpi_daily_renewable = calculate_kpi_renewable(building.building_info["id"], aggregate)
            fig_trend_renewable = plot_daily_kpi(kpi_daily_renewable["daily_kpi"].copy())
            plt.suptitle(f"Trend KPI giornalieri sulle rinnovabili per l'edificio {building.building_info['name']}",
                         fontsize=20, fontweight="bold")
            plt.tight_layout(rect=(0, 0.03, 1, 0.99))
            plt.subplots_adjust(right=1.1)
            plt.savefig(f"../../figures/kpi/trend_renewable_{building.building_info['name']}.png")

            fig_distribution_renewable = plot_kpi_distribution(kpi_daily_renewable["daily_kpi"])
            plt.suptitle(f"Distribuzione dei KPI giornalieri sulle rinnovabili per l'edificio {building.building_info['name']}",
                         fontsize=20,  fontweight="bold")
            plt.tight_layout(rect=(0, 0.03, 1, 0.99))
            plt.subplots_adjust(right=0.98)
            plt.savefig(f"../../figures/kpi/distribution_renewable_{building.building_info['name']}.png")

        df_score_user = calculate_performance_score_user(building.building_info["id"], aggregate)
        fig_radar_user = plot_radar_user(df_score_user)
        plt.suptitle(f"Radar plot dei Performance Score per l'utente {building.building_info['name']}", fontsize=20,
                     fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.savefig(f"../../figures/kpi/radar_user_{building.building_info['name']}.png")


if __name__ == "__main__":
    run_kpi("anguillara")
