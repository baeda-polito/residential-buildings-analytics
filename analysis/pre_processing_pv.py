from src.building import load_anguillara
from settings import PROJECT_ROOT
import matplotlib.pyplot as plt
import pandas as pd
import os

building_list = load_anguillara(mode="offline")

for building in building_list:
    if building.building_info["user_type"] != "consumer":
        data_cleaned = building.energy_meter.energy_meter_data_cleaned.copy()
        data_raw = building.energy_meter.energy_meter_data.copy()
        data_cleaned["timestamp"] = pd.to_datetime(data_cleaned["timestamp"])
        data_raw["timestamp"] = pd.to_datetime(data_raw["timestamp"])

        data_cleaned["hour"] = data_cleaned["timestamp"].dt.strftime("%H:%M")
        data_cleaned["date"] = data_cleaned["timestamp"].dt.date
        data_cleaned_pivot = data_cleaned.pivot_table(index="date", columns="hour", values="productionPower", dropna=False)

        data_raw["hour"] = data_raw["timestamp"].dt.strftime("%H:%M")
        data_raw["date"] = data_raw["timestamp"].dt.date
        data_raw_pivot = data_raw.pivot_table(index="date", columns="hour", values="productionPower", dropna=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data_raw_pivot, cmap="inferno")
        ax.set_xticks(range(0, 95, 4 * 4))
        ax.set_xticklabels(data_raw["hour"].iloc[:95:4 * 4], rotation=0, fontsize=10, ha="center")
        ax.set_yticks(range(0, len(data_raw_pivot), 4))
        ax.set_yticklabels(data_raw_pivot.index[::4], fontsize=10)
        ax.set_xlabel("Hour of the day", fontsize=12)
        ax.set_ylabel("Date", fontsize=12)
        cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.5)
        cbar.set_label("Power (W)", fontsize=12)
        plt.title(f"Carpet plot for {building.building_info['name']} ({building.building_info['user_type']})",
                  fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, "figures", "pv_pre_processing", f"{building.building_info['id']}_raw.png"))
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data_cleaned_pivot, cmap="inferno")
        ax.set_xticks(range(0, 95, 4 * 4))
        ax.set_xticklabels(data_cleaned["hour"].iloc[:95:4 * 4], rotation=0, fontsize=10, ha="center")
        ax.set_yticks(range(0, len(data_cleaned_pivot), 4))
        ax.set_yticklabels(data_cleaned_pivot.index[::4], fontsize=10)
        ax.set_xlabel("Hour of the day", fontsize=12)
        ax.set_ylabel("Date", fontsize=12)
        cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.5)
        cbar.set_label("Power (W)", fontsize=12)
        plt.title(f"Carpet plot for {building.building_info['name']} ({building.building_info['user_type']})",
                  fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, "figures", "pv_pre_processing", f"{building.building_info['id']}_cleaned.png"))
        plt.show()
        plt.close(fig)
