from src.building import load_anguillara, load_garda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


def plot_heatmap_nan(aggregate: str, date_start: str, date_end: str):
    """
    Grafica un heatmap con i valori mancanti per ogni edificio per ogni giorno.
    :param aggregate: il nome dell'aggregato ("anguillara" o "garda")
    :param date_start: data di inizio nel formato "YYYY-MM-DD"
    :param date_end: data di fine nel formato "YYYY-MM-DD"
    :return: l'heat map
    """

    building_list = []
    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()


    datetime_range = pd.date_range(start=date_start, end=date_end, freq="15min", tz="UTC")

    list_missing_values_power = []
    list_missing_values_production = []
    for building in building_list:
        data = building.energy_meter.energy_meter_data.copy()
        data.set_index("timestamp", inplace=True)
        data.index = pd.to_datetime(data.index, utc=True)
        data = data.reindex(datetime_range)

        # Group by date and calculate the number of missing values per each date
        missing_values_power = data["power"].isnull()
        missing_values_power = missing_values_power.groupby(missing_values_power.index.date).sum()
        missing_values_power.name = building.building_info["name"]
        list_missing_values_power.append(missing_values_power)

        if building.building_info["user_type"] != "consumer":
            missing_values_production = data["productionPower"].isnull()
            missing_values_production = missing_values_production.groupby(missing_values_production.index.date).sum()
            missing_values_production.name = building.building_info["name"]
            list_missing_values_production.append(missing_values_production)


    df_missing_values_power = pd.concat(list_missing_values_power, axis=1)
    df_missing_values_production = pd.concat(list_missing_values_production, axis=1)

    colors = ["#1d8348", "#ff6c00", "#b71c1c"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    norm = Normalize(vmin=0, vmax=96)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(df_missing_values_power.T, aspect="auto", cmap=cmap, norm=norm)

    # Set minor ticks on x-axis
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)

    x_ticks = np.arange(0, len(df_missing_values_power.index), step=14)
    x_labels = pd.to_datetime(df_missing_values_power.index[x_ticks]).strftime("%Y-%m-%d")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(df_missing_values_power.columns)))  # Offset y-ticks by 0.5 for centering
    ax.set_yticklabels(df_missing_values_power.columns, va="center")

    ax.set_xlabel("Data", fontsize=14)
    ax.set_ylabel("Edificio", fontsize=14)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.15, aspect=50)
    cbar.set_label("Numero di valori mancanti", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([0, 16, 32, 48, 64, 80, 96])

    plt.title(f"Numero di valori di potenza mancanti giornalieri per edificio per {aggregate.title()}", fontsize=20, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"../../figures/power_pre_processing/missing_values_power_{aggregate}.png", dpi=300)


    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(df_missing_values_production.T, aspect="auto", cmap=cmap, norm=norm)

    # Set minor ticks on x-axis
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)

    x_ticks = np.arange(0, len(df_missing_values_production.index), step=14)
    x_labels = pd.to_datetime(df_missing_values_production.index[x_ticks]).strftime("%Y-%m-%d")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(df_missing_values_production.columns)))  # Offset y-ticks by 0.5 for centering
    ax.set_yticklabels(df_missing_values_production.columns, va="center")

    ax.set_xlabel("Data", fontsize=14)
    ax.set_ylabel("Edificio", fontsize=14)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.15, aspect=50)
    cbar.set_label("Numero di valori mancanti", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([0, 16, 32, 48, 64, 80, 96])

    plt.title(f"Numero di valori di potenza mancanti giornalieri per edificio per {aggregate.title()}", fontsize=20,
              fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"../../figures/pv_pre_processing/missing_values_production_{aggregate}.png", dpi=300)
