from src.building import load_anguillara
import pandas as pd
import matplotlib.pyplot as plt


building_list = load_anguillara()

for building in building_list:
    data = building.energy_meter.data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["weekday"] = data["timestamp"].dt.weekday
    data["day_type"] = "Weekday"
    data.loc[data["weekday"] == 5, "day_type"] = "Saturday"
    data.loc[data["weekday"] == 6, "day_type"] = "Sunday"
    data["hour"] = data["timestamp"].dt.strftime("%H:%M")

    fig, ax = plt.subplots(figsize=(12, 5), ncols=3, nrows=1)
    legend_handles = []
    hour_labels = data["hour"].iloc[:95:4*4]
    groups = data.groupby("day_type")

    for i, (name, group) in enumerate(groups):
        axes = ax[i]
        day_groups = group.groupby(group["timestamp"].dt.date)
        centroid = group.groupby("hour")["Load"].median()
        for date, day in day_groups:
            load_profile_line, = axes.plot(day["hour"], day["Load"], color="grey", alpha=0.2, label="Load profile")
            axes.set_title(name)
            axes.set_xlabel("Hour of the day", fontsize=12)
            axes.set_ylabel("Power (W)", fontsize=12)
        centroid_line, = axes.plot(centroid.index, centroid.values, color="red", label="Centroid")
        axes.set_xticks(range(0, 95, 4 * 4))
        axes.set_xticklabels(hour_labels, rotation=0, fontsize=10, ha="center")

        legend_handles.append(load_profile_line)
        legend_handles.append(centroid_line)

    plt.suptitle(f"Daily load profiles for {building.building_info['name']}", fontsize=16)
    legend = fig.legend(legend_handles, ["Load profile", "Centroid"], loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=12, ncol=2, fancybox=True, shadow=True)
    plt.tight_layout(rect=(0, 0.08, 1, 0.98))
    plt.show()
    plt.close(fig)


for building in building_list:
    data = building.energy_meter.data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date
    data["hour"] = data["timestamp"].dt.strftime("%H:%M")

    data_pivot = data.pivot_table(index="date", columns="hour", values="Load")

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
    plt.title(f"Carpet plot for {building.building_info['name']}", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
