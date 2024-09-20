from src.anomaly_detection.mlp import MultiLayerPerceptron
from src.anomaly_detection.data_handler import DataHandler
from src.anomaly_detection.anomaly_detection_functions import get_anomaly_severity
from src.building import load_anguillara
from pvlib.location import Location
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import plotly.io as pio
import pandas as pd
import torch

building_list = load_anguillara()

for building in building_list:

    if building.building_info["user_type"] != "consumer":

        uuid = building.building_info["id"]
        aggregate = building.building_info["aggregate"]

        mlp = MultiLayerPerceptron(input_size=5, hidden_layers=[64, 64], output_size=1)
        mlp.load_state_dict(torch.load(f"../data/pv_add/models/{uuid}.pth"))

        location = Location(latitude=building.building_info["coordinates"][1], longitude=building.building_info["coordinates"][0])

        energy_data = building.energy_meter.energy_meter_data_cleaned
        energy_data["timestamp"] = pd.to_datetime(energy_data["timestamp"])
        weather_data = pd.read_csv("../data/weather/anguillara.csv")
        weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])
        solar_angles = location.get_solarposition(times=weather_data["timestamp"]).reset_index()
        weather_data = pd.merge(weather_data, solar_angles[["timestamp", "azimuth", "zenith"]], on="timestamp",
                                how="right")
        data_total = pd.merge(energy_data[["timestamp", "productionPower"]], weather_data, on="timestamp", how="left")
        data_total = data_total.rename(columns={"productionPower": "Production"})
        data_total.set_index("timestamp", inplace=True)
        data_total = data_total.dropna(subset=["air_temp", "ghi", "dni", "azimuth", "zenith"])
        # Replace NaN in Production column with zeros
        data_total["Production"] = data_total["Production"].fillna(0)
        data = data_total.copy()

        data_handler = DataHandler(energy_data=energy_data, weather_data=weather_data)
        X, y, x_scaler, y_scaler = data_handler.preprocess(data_total)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        y_pred = mlp(X_tensor).detach().numpy()

        y_pred_rescaled = y_scaler.inverse_transform(y_pred)
        y_true_rescaled = y_scaler.inverse_transform(y)

        residuals = y_pred_rescaled - y_true_rescaled

        df_add = pd.DataFrame({"y_true": y_true_rescaled.flatten(), "y_pred": y_pred_rescaled.flatten(), "residuals": residuals.flatten()}, index=data.index)
        df_add["hour"] = df_add.index.hour
        threshold = pd.read_csv(f"../data/pv_add/threshold/{uuid}.csv")
        df_add = pd.merge(df_add, threshold, on="hour", how="left")
        df_add['residuals'] = df_add['residuals'].clip(lower=0)
        df_add.index = data.index

        df_add["severity"] = df_add.apply(lambda x: get_anomaly_severity(x["y_true"], x["y_pred"], x["LT"], x["HT"]), axis=1)

        # Static heatmap of the severity
        df_add["date"] = df_add.index.date
        df_add['hour'] = df_add.index.strftime('%H:%M')

        pivot_table = df_add.pivot(index='date', columns='hour', values='severity')
        # Convert into float with one decimal
        pivot_table = pivot_table.astype(float).round(1)
        # Replace 0 with NaN, so they are not displayed
        pivot_table.replace(0, float('nan'), inplace=True)
        # Convert index in date index
        pivot_table.index = pd.to_datetime(pivot_table.index)
        # Extract the index as %b-%d
        pivot_table.index = pivot_table.index.strftime('%b-%d')

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Identificazione delle anomalie", fontsize=16, fontweight='bold')
        ax.set_ylabel("Ora del giorno", fontsize=14)
        ax.set_xlabel("Data", fontsize=14)
        cmap = clr.LinearSegmentedColormap.from_list('custom', ['#ffaa00', '#ff0000'], N=256)
        cax = ax.matshow(pivot_table.T, cmap=cmap, aspect='auto')

        # Set the x-axis labels one each 7
        ax.set_yticks(range(0, len(pivot_table.columns), 8))
        ax.set_yticklabels(pivot_table.columns[::8])
        # Extract the index as %b-%d
        ax.set_xticks(range(0, len(pivot_table.index), 14))
        ax.set_xticklabels(pivot_table.index[::14])

        # ax.set_xticklabels(pivot_table.index[::14])

        # Set xtickslabels position to bottom
        ax.xaxis.set_ticks_position('bottom')
        # Delete xticks major
        # ax.tick_params(axis='x', which='major', length=0)
        # ax.tick_params(axis='y', which='major', length=0)

        # Add minor ticks for gridlines
        ax.set_yticks([x - 0.5 for x in range(1, len(pivot_table.columns))], minor=True)
        ax.set_xticks([y - 0.5 for y in range(1, len(pivot_table.index))], minor=True)
        ax.grid(which='minor', color='#e1e1e1', linestyle='-', linewidth=1)

        # Add colorbar on the bottom, horizontally
        cbar = fig.colorbar(cax, orientation='horizontal', pad=0.1, aspect=50)
        cbar.set_label('Severità', fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks([0.5, 1])
        plt.tight_layout()
        plt.savefig(f"../figures/pv_evaluation/{uuid}_heatmap.png", dpi=300)

        # Count the number of 0.5 and 1 in the df_add severity
        count_low = df_add[df_add["severity"] == 0.5].shape[0]
        count_high = df_add[df_add["severity"] == 1].shape[0]

        # Sum the difference between residuals and LT when severity is 0.5
        sum_low = (df_add[df_add["severity"] == 0.5]["residuals"] - df_add[df_add["severity"] == 0.5]["LT"]).sum()*0.25/1000
        sum_high = (df_add[df_add["severity"] == 1]["residuals"] - df_add[df_add["severity"] == 1]["HT"]).sum()*0.25/1000

        print(f"{building.building_info['name']}:")
        print(f"Numeri di anomalie di bassa severità: {count_low} -- Energia associata: {sum_low:.1f} kWh")
        print(f"Numeri di anomalie di alta severità: {count_high} -- Energia associata: {sum_high:.1f} kWh")
        print("-"*50)
