import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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
    with open("config.json", "r") as f:
        config = json.load(f)

    kpi_columns = daily_kpi.columns.tolist()
    kpi_columns.remove("date")

    fig, axes = plt.subplots(1, len(kpi_columns), figsize=(4 * len(kpi_columns), 4))

    for i, kpi in enumerate(kpi_columns):
        ax = axes[i]
        ax.grid(axis="y", alpha=0.4)
        sns.histplot(daily_kpi[kpi], bins=20, color="skyblue", edgecolor="black", ax=ax, kde=True,
                     kde_kws={"bw_adjust": 0.5})
        ax.lines[0].set_color("red")
        ax.set_title(f"{config[kpi]['name']}", fontsize=16)
        ax.set_xlabel("Valore del KPI", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)

        if config[kpi]["uom"] == "Percentuale [%]":
            ax.set_xlim(0, 100)
        elif config[kpi]["uom"] == "Valore normalizzato [-]":
            ax.set_xlim(0, 1)

    return fig


def plot_radar_user(df_score: pd.DataFrame):
    """
    Radar plot per gli indicatori di performance di un utente
    :param df_score: il dataframe con i KPI e i relativi punteggi
    :return: la figura con il plot
    """
    with open("config.json", "r") as f:
        config = json.load(f)

    num_vars = len(df_score)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    values = df_score['value'].tolist()
    values += values[:1]

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, color='skyblue', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='skyblue', alpha=0.25)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([20, 40, 60, 80, 100], color="grey", fontsize=8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(df_score['name'], fontsize=12)

    for label, angle in zip(ax.get_xticklabels(), angles):
        if 0 <= angle < np.pi / 2:  # Right side
            label.set_horizontalalignment('left')
        elif angle == np.pi / 2 or angle == 3 * np.pi / 2:
            label.set_horizontalalignment('center')
        elif np.pi / 2 < angle < 3 * np.pi / 2:
            label.set_horizontalalignment('right')
        else:  # Bottom-right side
            label.set_horizontalalignment('left')
    ax.set_rlabel_position(180 / num_vars)

    return fig


def plot_boxplot_kpi_aggregate(df_kpi_aggregate: pd.DataFrame):
    """
    Boxplot dei KPI di ogni edificio all'interno dell'aggregato
    :param df_kpi_aggregate: il dataframe con i KPI di ogni edificio
    :return: la figura con il plot
    """

    with open("config.json", "r") as f:
        config = json.load(f)

    kpi_columns = df_kpi_aggregate.columns.tolist()
    kpi_columns.remove("date")
    kpi_columns.remove("building_name")

    num_buildings = df_kpi_aggregate["building_name"].nunique()

    if num_buildings > 5:
        size_param_hor = 1
        size_param_ver = len(kpi_columns)
    else:
        size_param_hor = 1.3
        size_param_ver = len(kpi_columns) * 1.2

    fig, axes = plt.subplots(len(kpi_columns), 1, figsize=(10 * size_param_hor, 3 * size_param_ver))

    for i, kpi in enumerate(kpi_columns):

        ax = axes[i]
        sns.boxplot(data=df_kpi_aggregate, x="building_name", y=kpi, ax=ax, color="skyblue")
        ax.set_title(f"{config[kpi]['name']}", fontsize=16)
        ax.set_xlabel("Building", fontsize=14)
        ax.set_ylabel(config[kpi]["uom"], fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        if config[kpi]["uom"] == "Percentuale [%]":
            ax.set_ylim(0, 100)
        elif config[kpi]["uom"] == "Valore normalizzato [-]":
            ax.set_ylim(0, 1)

    return fig


if __name__ == "__main__":
    from src.kpi.calculate import calculate_kpi_aggregate
    aggregate = "anguillara"

    df_kpi_load, df_kpi_flexibility, df_kpi_renewable = calculate_kpi_aggregate(aggregate)
    fig = plot_boxplot_kpi_aggregate(df_kpi_load)
    plt.suptitle(f"Boxplot dei KPI sul carico per l'aggregato di {aggregate.title()}", fontsize=20, fontweight="bold")
    plt.tight_layout(rect=(0, 0.03, 1, 0.99))
    plt.savefig(f"../../figures/kpi/boxplot_load_{aggregate}.png")

    fig = plot_boxplot_kpi_aggregate(df_kpi_flexibility)
    plt.suptitle(f"Boxplot dei KPI sulla flessibilità per l'aggregato di {aggregate.title()}", fontsize=20, fontweight="bold")
    plt.tight_layout(rect=(0, 0.03, 1, 0.99))
    plt.savefig(f"../../figures/kpi/boxplot_flexibility_{aggregate}.png")

    fig = plot_boxplot_kpi_aggregate(df_kpi_renewable)
    plt.suptitle(f"Boxplot dei KPI sulle rinnovabili per l'aggregato di {aggregate.title()}", fontsize=20, fontweight="bold")
    plt.tight_layout(rect=(0, 0.03, 1, 0.99))
    plt.savefig(f"../../figures/kpi/boxplot_renewable_{aggregate}.png")

