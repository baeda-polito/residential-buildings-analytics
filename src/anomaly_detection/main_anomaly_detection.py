import numpy as np
import pandas as pd
from src.anomaly_detection.mlp import MultiLayerPerceptron
from src.anomaly_detection.evaluate import predict
from src.anomaly_detection.anomaly_detection_functions import get_anomaly_severity
import torch
import pickle


def detect_anomaly(uuid: str, ghi: float, dni: float, air_temp: float, azimuth: float,
                   zenith: float, y_true: float, hour: int):
    """
    Funzione che calcola la produzione fotovoltaica attesa per un edificio in un determinato istante temporale e
    confronta il valore reale con quello previsto dal modello. Dopo di ciò, lo confronta con
    i threhsold calcolati in precedenza per determinare se vi è un'anomalia.
    :param uuid: id dell'edificio
    :param ghi: irraggiamento globale orizzontale
    :param dni: irraggiamento diretto normale
    :param air_temp: temperatura dell'aria
    :param azimuth: angolo di azimut del sole
    :param zenith: angolo di zenit del sole
    :param y_true: valore reale di produzione fotovoltaica
    :param hour: ora del giorno in intero (0-23)
    :return: booleano che indica se vi è un'anomalia, severità dell'anomalia, valore predetto
    """

    mlp = MultiLayerPerceptron(input_size=5, hidden_layers=[64, 64], output_size=1)
    mlp.load_state_dict(torch.load(f"../../data/pv_add/models/{uuid}.pth"))

    # Load the scaler
    x_scaler = pickle.load(open(f"../../data/pv_add/scalers/x_scaler_{uuid}.pkl", "rb"))
    y_scaler = pickle.load(open(f"../../data/pv_add/scalers/y_scaler_{uuid}.pkl", "rb"))

    x = np.array([ghi, dni, air_temp, azimuth, zenith]).reshape(1, -1)
    x_norm = x_scaler.transform(x)

    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_pred = predict(mlp, x_tensor).detach().numpy()
    y_pred_rescaled = y_scaler.inverse_transform(y_pred)[0][0]

    # Load the threshold
    threshold = pd.read_csv(f"../../data/pv_add/threshold/{uuid}.csv")

    lt = threshold.loc[threshold["hour"] == hour, "LT"].values[0]
    ht = threshold.loc[threshold["hour"] == hour, "HT"].values[0]

    severity = get_anomaly_severity(y_true, y_pred_rescaled, lt, ht)
    anomaly_bool = severity > 0

    return anomaly_bool, severity, y_pred_rescaled
