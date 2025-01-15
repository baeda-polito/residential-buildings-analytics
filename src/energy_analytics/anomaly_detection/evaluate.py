import os
import json
import torch
import numpy as np
import pandas as pd
from pvlib.location import Location
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from settings import PROJECT_ROOT
from ..building import Building
from .mlp import MultiLayerPerceptron
from .data_handler import DataHandler
from .viz import plot_predictions, plot_distribution, plot_pred_vs_true


def predict(model: MultiLayerPerceptron, X_tensor: torch.Tensor):
    """
    Funzione che effettua la predizione della produzione fotovoltaica per un tensore di dati X.
    Nello step di previsione, controlla che i valori di irradianza non siano nulli, in tal caso predice un valore nullo.
    Inoltre, controlla che i valori predetti non siano negativi, in tal caso li imposta a 0.

    Args:
        model (MultiLayerPerceptron): modello di regressione trainato
        X_tensor (torch.Tensor): tensor con i dati di input

    Returns:
        torch.Tensor: tensor con i valori predetti
    """

    model.eval()

    mask_zero_irradiance = (X_tensor[:, 0] == 0) & (X_tensor[:, 1] == 0)

    if mask_zero_irradiance.any():
        output = torch.zeros(X_tensor.size(0), 1)
        output[~mask_zero_irradiance] = model(X_tensor[~mask_zero_irradiance])  # Only forward pass for rows where the condition is False
    else:
        output = model(X_tensor)

    # Clip the output to ensure no negative values (clipping to zero)
    output = torch.clamp(output, min=0)

    return output


def calc_metrics(y_true: np.array, y_pred: np.array):
    """
    Funzione che calcola le metriche di errore (MAE, RMSE, R2, MAPE, MSE) tra i valori veri e quelli predetti. Viene calcolato
    solo per i valori diversi da 0 (quindi solo per i valori in cui c'è produzione fotovoltaica), per evitare di modificare
    la media con errori nulli (poiché il modello predice automaticamente 0 se non vi è irradianza)

    Args:
        y_true (np.array): array con i valori di produzione reali
        y_pred (np.array): array con i valori di produzione predetti

    Returns:
        dict: dizionario con le metriche calcolate
    """

    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "MAE": np.round(mae, 2),
        "RMSE": np.round(rmse, 2),
        "R2": np.round(r2, 2),
        "MAPE": np.round(mape * 100, 2),
        "MSE": np.round(mse, 2)
    }


def evaluate_pv_model(building: Building):
    """
    Valuta il modello allenato di previsione della produzione fotovoltaica di un edificio. Calcola le metriche di errore
    (MAE, RMSE, R2, MAPE, MSE) tra i valori veri e quelli predetti, e salva i risultati in un file json. Inoltre, crea
    dei grafici per visualizzare i risultati. In particolare, crea un grafico con le serie temporali dei valori predetti
     vs reali, un grafico con la distribuzione degli errori e un grafico con la distribuzione dei valori predetti vs
     reali. Salva i grafici in formato html.

    """

    mlp = MultiLayerPerceptron(input_size=5, hidden_layers=[64, 64], output_size=1)
    mlp.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", f"{building.building_info['id']}.pth")))

    location = Location(latitude=building.building_info["coordinates"][1],
                        longitude=building.building_info["coordinates"][0])

    energy_data = building.energy_data.data
    weather_data = building.energy_data.weather_data
    data_handler = DataHandler(energy_data=energy_data, weather_data=weather_data)
    data = data_handler.create_data(location=location)
    X, y, x_scaler, y_scaler = data_handler.preprocess(data, building.building_info["id"], save_scalers=False)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    y_pred = predict(mlp, X_tensor).detach().numpy()
    y_pred_train = predict(mlp, X_train_tensor).detach().numpy()
    y_pred_val = predict(mlp, X_val_tensor).detach().numpy()

    y_pred_rescaled = y_scaler.inverse_transform(y_pred)
    y_true_rescaled = y_scaler.inverse_transform(y)

    y_pred_train_rescaled = y_scaler.inverse_transform(y_pred_train)
    y_true_train_rescaled = y_scaler.inverse_transform(y_train)

    y_pred_val_rescaled = y_scaler.inverse_transform(y_pred_val)
    y_true_val_rescaled = y_scaler.inverse_transform(y_val)

    metrics_train = calc_metrics(y_true_train_rescaled, y_pred_train_rescaled)
    with open(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "metrics", f"train_{building.building_info['id']}"), "w") as f:
        json.dump(metrics_train, f)

    metrics_val = calc_metrics(y_true_val_rescaled, y_pred_val_rescaled)
    with open(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "metrics", f"val_{building.building_info['id']}"), "w") as f:
        json.dump(metrics_val, f)

    # Figure true vs pred
    data_plot = pd.DataFrame({"true": y_true_rescaled.flatten(), "pred": y_pred_rescaled.flatten()},
                             index=data.index)
    fig = plot_predictions(data_plot, building.building_info["name"])
    fig.write_html(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection", "pv_models", f"{building.building_info['id']}_pred.html"))

    # Residuals analysis
    residuals = y_pred_rescaled - y_true_rescaled
    data_residuals = data.copy()
    data_residuals["residuals"] = residuals

    fig_dist = plot_distribution(data_residuals[(data_residuals['ghi'] != 0) & (data_residuals['dni'] != 0)], building.building_info["name"])
    fig_dist.write_html(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection", "pv_models", f"{building.building_info['id']}_dist.html"))

    fig_true_vs_pred = plot_pred_vs_true(data_plot, building.building_info["name"])
    fig_true_vs_pred.write_html(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection", "pv_models", f"{building.building_info['id']}_true_vs_pred.html"))
