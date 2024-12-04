from src.anomaly_detection.mlp import MultiLayerPerceptron
from src.anomaly_detection.data_handler import DataHandler
from src.building import Building
from pvlib.location import Location
import pandas as pd
import torch


def calculate_threshold(uuid: str, aggregate: str):
    """
    Funzione che calcola i threshold per la rilevazione delle anomalie per un edificio. I threshold vengono calcolati
    come la media più 2 (low threshold) o 3 (high threshold) deviazioni standard dei residui del modello di previsione
    della produzione fotovoltaica. I threshold vengono calcolati per ogni ora del giorno e salvati in un file csv
    chiamato threhsold_{uuid}.csv.
    :param uuid: l'id dell'edificio
    :param aggregate: il nome dell'aggregato ("anguillara" o "garda")
    :return: None
    """

    mlp = MultiLayerPerceptron(input_size=5, hidden_layers=[64, 64], output_size=1)
    mlp.load_state_dict(torch.load(f"../../data/pv_add/models/{uuid}.pth"))

    building = Building(uuid=uuid, aggregate=aggregate)
    location = Location(latitude=building.building_info["coordinates"][1], longitude=building.building_info["coordinates"][0])

    energy_data = building.energy_meter.data
    weather_data = pd.read_csv("../../data/weather/anguillara.csv")
    data_handler = DataHandler(energy_data=energy_data, weather_data=weather_data)
    data = data_handler.create_data(location=location)
    X, y, x_scaler, y_scaler = data_handler.preprocess(data, uuid, save_scalers=False)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    y_pred = mlp(X_tensor).detach().numpy()

    y_pred_rescaled = y_scaler.inverse_transform(y_pred)
    y_true_rescaled = y_scaler.inverse_transform(y)

    residuals = y_pred_rescaled - y_true_rescaled
    df_residuals = pd.DataFrame(residuals, columns=["residuals"], index=data.index)

    df_residuals["hour"] = df_residuals.index.hour
    df_residuals = df_residuals[df_residuals["residuals"] >= 0]

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

    df_threshold.to_csv(f"../../data/pv_add/threshold/{uuid}.csv", index=False)


def get_anomaly_threhsold(y_pred, lt, ht):
    """
    Funzione che calcola i threshold per la rilevazione delle anomalie per un'osservazione. I threshold vengono calcolati
    come la differenza tra il valore predetto e il low threshold (lt) o il high threshold (ht).
    :param y_pred: il valore predetto
    :param lt: il low threshold
    :param ht: il high threshold
    :return: lower_threshold, higher_threshold
    """

    lower_threshold = y_pred - lt
    higher_threshold = y_pred - ht

    return lower_threshold, higher_threshold


def get_anomaly_severity(y_true, y_pred, lt, ht):
    """
    Funzione che calcola la severità dell'anomalia per una predizione. La severità è calcolata come segue:
    - 1 se il valore osservato è maggiore di quello predetto più il high threshold
    - 0.5 se il valore osservato è compreso tra il il valore predetto più il low threshold e il valore predetto più il
    high threshold
    - 0 nel resto dei casi
    :param y_true: il valore osservato
    :param y_pred: il valore predetto
    :param lt: il low threshold
    :param ht: il high threshold
    :return: il valore di severità dell'anomalia
    """

    if (y_pred - lt) > y_true > (y_pred - ht):
        return 0.5
    elif y_true < (y_pred - ht):
        return 1
    else:
        return 0
