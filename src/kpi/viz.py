import matplotlib.pyplot as plt
import pandas as pd
import json


def plot_daily_kpi(daily_kpi: pd.DataFrame):
    """
    Calendar plot dei KPI giornalieri
    :param daily_kpi:
    :return:
    """

    with open("config.json", "r") as f:
        config = json.load(f)

    kpi_columns = daily_kpi.columns.tolist()
    kpi_columns.remove("date")

    daily_kpi["date"] = pd.to_datetime(daily_kpi["date"])
    daily_kpi["week"] = daily_kpi["date"].dt.isocalendar().week
    daily_kpi["day_of_week"] = daily_kpi["date"].dt.dayofweek

    fig, axes = plt.subplots(len(kpi_columns), 1, figsize=(20, 3 * len(kpi_columns)))

    for i, kpi in enumerate(kpi_columns):
        ax = axes[i]
        daily_kpi_pivot = daily_kpi.pivot(index="day_of_week", columns="week", values=kpi)
        daily_kpi_pivot = daily_kpi_pivot.reindex(index=[0, 1, 2, 3, 4, 5, 6])
        daily_kpi_pivot = daily_kpi_pivot.reindex(columns=range(1, 53))

        if config[kpi]["uom"] == "Percentuale [%]":
            im = ax.imshow(daily_kpi_pivot, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        elif config[kpi]["uom"] == "Valore normalizzato [-]":
            im = ax.imshow(daily_kpi_pivot, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        else:
            im = ax.imshow(daily_kpi_pivot, cmap="RdYlGn", aspect="auto")
        ax.set_title(f"{config[kpi]['name']}", fontsize=16)
        ax.set_xlabel("Week of the year", fontsize=14)
        ax.set_ylabel("Day of the week", fontsize=14)

        ax.set_xticks(range(0, 52, 4))
        ax.set_xticklabels(range(1, 53, 4))

        ax.set_yticks(range(0, 7))
        ax.set_yticklabels(["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(config[kpi]["uom"], fontsize=14)

    return fig


def plot_kpi_distribution(daily_kpi: pd.DataFrame):
    """
    Plot della distribuzione dei KPI giornalieri per un edificio
    :param daily_kpi: il dataframe con i KPI giornalieri. Ha una colonna "date" e una colonna per ogni KPI
    :return: la figura con i plot
    """
    pass




if __name__ == "__main__":
    from src.kpi.calculate import calculate_kpi_load, calculate_kpi_flexibility, calculate_kpi_renewable
    user_id = "7436df46-294b-4c97-bd1b-8aaa3aed97c5"
    user_name = "DU_1"
    aggregate = "anguillara"
    kpi = calculate_kpi_renewable(user_id, aggregate)
    fig = plot_daily_kpi(kpi["daily_kpi"])
    plt.suptitle(f"Trend KPI giornalieri sulle rinnovabili per l'edificio {user_name}", fontsize=20,
                 fontweight="bold")
    plt.tight_layout(rect=(0, 0.03, 1, 0.99))
    fig.subplots_adjust(right=1.1)
    plt.show()
