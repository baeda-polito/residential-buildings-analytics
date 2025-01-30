import os
import torch
import pandas as pd
from pvlib.location import Location

from settings import PROJECT_ROOT
from .mlp import MultiLayerPerceptron
from .data_handler import DataHandler
from ..building import Building


def calculate_threshold(building: Building):
    """
    Funzione che calcola i threshold per la rilevazione delle anomalie per un edificio. I threshold vengono calcolati
    come la media più 2 (low threshold) o 3 (high threshold) deviazioni standard dei residui del modello di previsione
    della produzione fotovoltaica. I threshold vengono calcolati per ogni ora del giorno e salvati in un file csv
    chiamato threhsold_{uuid}.csv.

    Args:
        building (Building): oggetto Building con le informazioni sull'edificio.

    Returns:
        None
    """

    mlp = MultiLayerPerceptron(input_size=5, hidden_layers=[64, 64], output_size=1)
    mlp.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", f"{building.building_info['id']}.pth")))

    location = Location(latitude=building.building_info["coordinates"][1],
                        longitude=building.building_info["coordinates"][0])

    energy_data = building.energy_data.data
    weather_data = building.energy_data.weather_data
    data_handler = DataHandler(energy_data=energy_data, weather_data=weather_data)
    data = data_handler.create_data(location=location)
    X, y, x_scaler, y_scaler = data_handler.preprocess(data, building.building_info['id'], save_scalers=False)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    y_pred = mlp(X_tensor).detach().numpy()

    y_pred_rescaled = y_scaler.inverse_transform(y_pred)
    y_true_rescaled = y_scaler.inverse_transform(y)

    residuals = y_pred_rescaled - y_true_rescaled
    df_residuals = pd.DataFrame(residuals, columns=["residuals"], index=data.index)

    df_residuals["hour"] = df_residuals.index.hour
    df_residuals = df_residuals[df_residuals["residuals"] >= 0]

    index_irradiance = weather_data.loc[(weather_data["ghi"] > 0) | (weather_data["dni"] > 0), "timestamp"]

    df_residuals = df_residuals.loc[df_residuals.index.isin(index_irradiance)]

    lt_dict = {}
    ht_dict = {}
    # Calculate the Z-score threshold for each hour
    for hour, group in df_residuals.groupby("hour"):
        residuals_hour = group["residuals"]
        high_threshold = residuals_hour.mean() + 3 * residuals_hour.std()
        low_threshold = residuals_hour.mean() + 2 * residuals_hour.std()
        lt_dict[hour] = low_threshold.round(2)
        ht_dict[hour] = high_threshold.round(2)

    df_threshold = pd.DataFrame({"LT": lt_dict, "HT": ht_dict})
    df_threshold.reset_index(inplace=True, names=["hour"])

    df_threshold.to_csv(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "thresholds", f"threshold_{building.building_info['id']}.csv"), index=False)


def get_anomaly_threhsold(y_pred: float, lt: float, ht: float):
    """
    Funzione che calcola i threshold per la rilevazione delle anomalie per un'osservazione. I threshold vengono calcolati
    come la differenza tra il valore predetto e il low threshold (lt) o il high threshold (ht).

    Args:
        y_pred (float): il valore predetto
        lt (float): il low threshold
        ht (float): il high threshold

    Returns:
        lower_threshold (float): il lower threshold
        higher_threshold (float): il higher threshold
    """

    lower_threshold = y_pred - lt
    higher_threshold = y_pred - ht

    return lower_threshold, higher_threshold


def get_anomaly_severity(y_true: float, y_pred: float, lt: float, ht: float):
    """
    Funzione che calcola la severità dell'anomalia per una predizione. La severità è calcolata come segue:
    - 1 se il valore osservato è maggiore di quello predetto più il high threshold
    - 0.5 se il valore osservato è compreso tra il il valore predetto più il low threshold e il valore predetto più il
    high threshold
    - 0 nel resto dei casi

    Args:
        y_true (float): il valore osservato
        y_pred (float): il valore predetto
        lt (float): il low threshold
        ht (float): il high threshold

    Returns:
        severity (float): la severità dell'anomalia
    """

    if (y_pred - lt) > y_true > (y_pred - ht):
        return 0.5
    elif y_true < (y_pred - ht):
        return 1
    else:
        return 0
