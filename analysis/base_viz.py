import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.building import load_anguillara, load_garda
from settings import PROJECT_ROOT


building_list = load_garda()

for building in building_list:
    data = building.energy_meter.data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["weekday"] = data["timestamp"].dt.weekday
    data["day_type"] = "Weekday"
    data.loc[data["weekday"] == 5, "day_type"] = "Saturday"
    data.loc[data["weekday"] == 6, "day_type"] = "Sunday"
    data["hour"] = data["timestamp"].dt.strftime("%H:%M")

    fig, ax = plt.subplots(figsize=(12, 5), ncols=3, nrows=1, sharey='all')
    legend_handles = []
    hour_labels = data["hour"].iloc[:95:4*4]
    groups = data.groupby("day_type")

    for i, (name, group) in enumerate(groups):
        axes = ax[i]
        day_groups = group.groupby(group["timestamp"].dt.date)
        centroid = group.groupby("hour")["Net"].mean()
        for date, day in day_groups:
            load_profile_line, = axes.plot(day["hour"], day["Net"], color="grey", alpha=0.2, label="Load profile")
            axes.set_title(name)
            axes.set_xlabel("Hour of the day", fontsize=12)
            axes.set_ylabel("Power (W)", fontsize=12)
        centroid_line, = axes.plot(centroid.index, centroid.values, color="red", label="Centroid")
        axes.set_xticks(range(0, 95, 4 * 4))
        axes.set_xticklabels(hour_labels, rotation=0, fontsize=10, ha="center")

        legend_handles.append(load_profile_line)
        legend_handles.append(centroid_line)

    plt.suptitle(f"Daily load profiles for {building.building_info['name']} ({building.building_info['user_type']})",
                 fontsize=16)
    legend = fig.legend(legend_handles, ["Load profile", "Centroid"], loc="lower center", bbox_to_anchor=(0.5, 0.01),
                        fontsize=12, ncol=2, fancybox=True, shadow=True)
    plt.tight_layout(rect=(0, 0.08, 1, 0.98))
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "load_profiles", f"{building.building_info['id']}.png"))
    plt.show()
    plt.close(fig)


for building in building_list:
    data = building.energy_meter.data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    time_range = pd.date_range(start="2024-03-01T00:00:00Z", end=data["timestamp"].max().strftime('%Y-%m-%dT%H:%M:%SZ'), freq="15min")
    data = data.merge(pd.DataFrame(time_range, columns=["timestamp"]), on="timestamp", how="right")
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date
    data["hour"] = data["timestamp"].dt.strftime("%H:%M")

    data_pivot = data.pivot_table(index="date", columns="hour", values="Load", dropna=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_pivot, cmap="Spectral_r")
    ax.set_xticks(range(0, 95, 4 * 4))
    ax.set_xticklabels(data["hour"].iloc[:95:4*4], rotation=0, fontsize=10, ha="center")
    ax.set_yticks(range(0, len(data_pivot), 4))
    ax.set_yticklabels(data_pivot.index[::4], fontsize=10)
    ax.set_xlabel("Hour of the day", fontsize=12)
    ax.set_ylabel("Date", fontsize=12)
    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.5)
    cbar.set_label("Power (W)", fontsize=12)
    plt.title(f"Carpet plot for {building.building_info['name']} ({building.building_info['user_type']})",
              fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "carpet_plot", f"{building.building_info['id']}_cleaned.png"))
    plt.show()
    plt.close(fig)

for building in building_list:
    data = building.energy_meter.energy_meter_data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    time_range = pd.date_range(start="2024-03-01T00:00:00Z", end=data["timestamp"].max().strftime('%Y-%m-%dT%H:%M:%SZ'),
                               freq="15min")
    data = data.merge(pd.DataFrame(time_range, columns=["timestamp"]), on="timestamp", how="right")
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date
    data["hour"] = data["timestamp"].dt.strftime("%H:%M")
    if building.building_info["user_type"] == "consumer":
        data["productionPower"] = 0
    data["Load"] = np.where(data["power"] < 0, data["productionPower"] - abs(data["power"]),
                        data["productionPower"] + data["power"])
    data["Load"][data["Load"] < 0] = np.nan
    data_pivot = data.pivot_table(index="date", columns="hour", values="Load", dropna=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_pivot, cmap="Spectral_r")
    ax.set_xticks(range(0, 95, 4 * 4))
    ax.set_xticklabels(data["hour"].iloc[:95:4*4], rotation=0, fontsize=10, ha="center")
    ax.set_yticks(range(0, len(data_pivot), 4))
    ax.set_yticklabels(data_pivot.index[::4], fontsize=10)
    ax.set_xlabel("Hour of the day", fontsize=12)
    ax.set_ylabel("Date", fontsize=12)
    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.5)
    cbar.set_label("Power (W)", fontsize=12)
    plt.title(f"Carpet plot for {building.building_info['name']} ({building.building_info['user_type']})",
              fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "carpet_plot", f"{building.building_info['id']}_raw.png"))
    plt.show()
    plt.close(fig)


for building in building_list:
    data = building.energy_meter.energy_meter_data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    if building.building_info["user_type"] != "consumer":
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(data["timestamp"], data["power"], color="blue", label="Power")
        ax.plot(data["timestamp"], data["impEnergy"], color="green", label="Imported energy")
        ax.plot(data["timestamp"], data["expEnergy"], color="red", label="Exported energy")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("[Wh]", fontsize=12)
        ax.set_xlim(data["timestamp"].min(), data["timestamp"].max())
        ax.set_title(f"{building.building_info['name']} ({building.building_info['user_type']})", fontsize=18)
        fig.legend(bbox_to_anchor=(0.5, 0), loc='lower center', fontsize=12, ncol=3, fancybox=True, shadow=True)
        plt.tight_layout(rect=(0, 0.08, 1, 1))
        plt.savefig(os.path.join(PROJECT_ROOT, "figures", "misc", f"{building.building_info['id']}_imp_exp.png"))
        plt.show()
        plt.close(fig)
