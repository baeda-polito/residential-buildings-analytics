import os
import torch
import pickle
import numpy as np
import pandas as pd
from loguru import logger

from settings import PROJECT_ROOT
from .mlp import MultiLayerPerceptron
from .evaluate import predict
from .anomaly_detection_functions import get_anomaly_severity


def detect_anomaly(uuid: str, ghi: float, dni: float, air_temp: float, azimuth: float,
                   zenith: float, y_true: float, hour: int):
    """
    Funzione che calcola la produzione fotovoltaica attesa per un edificio in un determinato istante temporale e
    confronta il valore reale con quello previsto dal modello. Dopo di ciò, lo confronta con
    i threhsold calcolati in precedenza per determinare se vi è un'anomalia.

    Args:
        uuid (str): id dell'edificio
        ghi (float): irraggiamento globale orizzontale
        dni (float): irraggiamento diretto normale
        air_temp (float): temperatura dell'aria
        azimuth (float): angolo di azimut del sole
        zenith (float): angolo di zenit del sole
        y_true (float): valore reale di produzione fotovoltaica
        hour (int): ora del giorno in intero (0-23)

    Returns:
        bool: booleano che indica se vi è un'anomalia
    """

    mlp = MultiLayerPerceptron(input_size=5, hidden_layers=[64, 64], output_size=1)
    mlp.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", f"{uuid}.pth")))

    try:
        x_scaler = pickle.load(open(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "scalers", f"x_scaler_{uuid}"), "rb"))
        y_scaler = pickle.load(open(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "scalers", f"y_scaler_{uuid}"), "rb"))
    except FileNotFoundError:
        logger.error("Scaler non trovati. Lancia la pipeline di addestramento prima di eseguire la rilevazione delle anomalie.")

    x = np.array([ghi, dni, air_temp, azimuth, zenith]).reshape(1, -1)
    x_norm = x_scaler.transform(x)

    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_pred = predict(mlp, x_tensor).detach().numpy()
    y_pred_rescaled = y_scaler.inverse_transform(y_pred)[0][0]

    try:
        threshold = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "thresholds", f"{uuid}.csv"))
    except FileNotFoundError:
        logger.error("Threshold non trovati. Lancia la pipeline di addestramento prima di eseguire la rilevazione delle anomalie.")

    lt = threshold.loc[threshold["hour"] == hour, "LT"].values[0]
    ht = threshold.loc[threshold["hour"] == hour, "HT"].values[0]

    severity = get_anomaly_severity(y_true, y_pred_rescaled, lt, ht)
    anomaly_bool = severity > 0

    return anomaly_bool, severity, y_pred_rescaled
