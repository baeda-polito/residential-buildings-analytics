from src.kpi.viz import plot_daily_kpi
from src.building import load_anguillara, load_garda
from src.kpi.calculate import calculate_kpi_load, calculate_kpi_flexibility, calculate_kpi_renewable
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
        fig = plot_daily_kpi(kpi_daily_load["daily_kpi"])
        plt.suptitle(f"Trend KPI giornalieri sul carico per l'edificio {building.building_info['name']}", fontsize=20,
                     fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.subplots_adjust(right=1.1)
        plt.savefig(f"../../figures/kpi/trend_energy_{building.building_info['name']}.png")

        kpi_daily_flexibility = calculate_kpi_flexibility(building.building_info["id"], aggregate)
        fig = plot_daily_kpi(kpi_daily_flexibility["daily_kpi"])
        plt.suptitle(f"Trend KPI giornalieri sulla flessibilità per l'edificio {building.building_info['name']}", fontsize=20,
                     fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.99))
        plt.subplots_adjust(right=1.1)
        plt.savefig(f"../../figures/kpi/trend_flexibility_{building.building_info['name']}.png")

        if building.building_info["user_type"] != "consumer":
            kpi_daily_renewable = calculate_kpi_renewable(building.building_info["id"], aggregate)
            fig = plot_daily_kpi(kpi_daily_renewable["daily_kpi"])
            plt.suptitle(f"Trend KPI giornalieri sulle rinnovabili per l'edificio {building.building_info['name']}", fontsize=20,
                         fontweight="bold")
            plt.tight_layout(rect=(0, 0.03, 1, 0.99))
            fig.subplots_adjust(right=1.1)
            plt.savefig(f"../../figures/kpi/trend_renewable_{building.building_info['name']}.png")


if __name__ == "__main__":
    run_kpi("anguillara")
