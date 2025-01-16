import os
import torch
import numpy as np
import pandas as pd
from pvlib.location import Location

from settings import PROJECT_ROOT
from ..building import Building
from .mlp import MultiLayerPerceptron
from .data_handler import DataHandler
from .anomaly_detection_functions import get_anomaly_severity


def get_anomaly_residuals(building: Building) -> pd.DataFrame:
    """
    Crea un dataframe dove sono presenti i valori reali, predetti, i residui e i threhsold, ogni 15 minuti.

    Args:
        building (Building): oggetto Building con i dati dell'edificio.

    Returns:
        pd.DataFrame: dataframe con le colonne 'y_true', 'y_pred', 'residuals', 'hour', 'LT', 'HT', 'severity'
    """

    mlp = MultiLayerPerceptron(input_size=5, hidden_layers=[64, 64], output_size=1)
    mlp.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", f"{building.building_info['id']}.pth")))

    location = Location(latitude=building.building_info["coordinates"][1],
                        longitude=building.building_info["coordinates"][0])

    energy_data = building.energy_data.data_cleaned
    energy_data["timestamp"] = pd.to_datetime(energy_data["timestamp"])
    weather_data = building.energy_data.weather_data
    weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])

    solar_angles = location.get_solarposition(times=weather_data["timestamp"]).reset_index()
    weather_data = pd.merge(weather_data, solar_angles[["timestamp", "azimuth", "zenith"]], on="timestamp",
                            how="right")

    data_total = pd.merge(energy_data[["timestamp", "productionPower"]], weather_data, on="timestamp", how="left")
    data_total = data_total.rename(columns={"productionPower": "Production"})
    data_total.set_index("timestamp", inplace=True)
    data_total = data_total.dropna(subset=["air_temp", "ghi", "dni", "azimuth", "zenith"])
    data = data_total.copy()

    data_handler = DataHandler(energy_data=energy_data, weather_data=weather_data)
    X, y, x_scaler, y_scaler = data_handler.preprocess(data_total, building.building_info["id"], save_scalers=False)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_pred = mlp(X_tensor).detach().numpy()
    y_pred_rescaled = y_scaler.inverse_transform(y_pred)
    y_true_rescaled = y_scaler.inverse_transform(y)

    residuals = y_pred_rescaled - y_true_rescaled

    df_add = pd.DataFrame({"y_true": y_true_rescaled.flatten(),
                           "y_pred": y_pred_rescaled.flatten(),
                           "residuals": residuals.flatten()},
                          index=data.index)

    df_add["hour"] = df_add.index.hour
    threshold = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "thresholds", f"threshold_{building.building_info['id']}.csv"))
    df_add = pd.merge(df_add, threshold, on="hour", how="left")
    df_add['residuals'] = df_add['residuals'].clip(lower=0)
    df_add.index = data.index

    df_add["severity"] = df_add.apply(lambda x: get_anomaly_severity(x["y_true"], x["y_pred"], x["LT"], x["HT"]), axis=1)
    df_add.loc[df_add["y_true"].isna(), :] = np.nan

    return df_add
