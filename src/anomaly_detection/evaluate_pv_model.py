from src.anomaly_detection.mlp import MultiLayerPerceptron
from src.anomaly_detection.data_handler import DataHandler
from src.anomaly_detection.viz import plot_predictions, plot_distribution, plot_pred_vs_true
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from pvlib.location import Location
from src.building import Building
import plotly.io as pio
import pandas as pd
import numpy as np
import torch
import json


def calc_metrics(y_true, y_pred):

    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "MAE": np.round(mae, 2),
        "RMSE": np.round(rmse, 2),
        "R2": np.round(r2, 2),
        "MAPE": np.round(mape * 100, 2)
    }


def evaluate_pv_model(uuid: str, aggregate: str = "anguillara"):

    mlp = MultiLayerPerceptron(input_size=5, hidden_layers=[64, 64], output_size=1)
    mlp.load_state_dict(torch.load(f"../../data/pv_add/models/{uuid}.pth"))

    building = Building(uuid=uuid, aggregate=aggregate)
    location = Location(latitude=building.building_info["coordinates"][1], longitude=building.building_info["coordinates"][0])

    energy_data = building.energy_meter.data
    weather_data = pd.read_csv("../../data/weather/anguillara.csv")
    data_handler = DataHandler(energy_data=energy_data, weather_data=weather_data)
    data = data_handler.create_data(location=location)
    X, y, x_scaler, y_scaler = data_handler.preprocess(data)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    y_pred = mlp(X_tensor).detach().numpy()

    y_pred_rescaled = y_scaler.inverse_transform(y_pred)
    y_true_rescaled = y_scaler.inverse_transform(y)

    metrics = calc_metrics(y_true_rescaled, y_pred_rescaled)
    with open(f"../../data/pv_add/metrics/{uuid}.json", "w") as f:
        json.dump(metrics, f)

    # Figure true vs pred
    data_plot = pd.DataFrame({"true": y_true_rescaled.flatten(), "pred": y_pred_rescaled.flatten()},
                             index=data.index)
    fig = plot_predictions(data_plot, building.building_info["name"])
    fig.write_html(f"../../figures/pv_evaluation/{uuid}_pred.html")

    # Residuals analysis
    residuals = y_pred_rescaled - y_true_rescaled
    data_residuals = data.copy()
    data_residuals["residuals"] = residuals

    fig_dist = plot_distribution(data_residuals[(data_residuals['ghi'] != 0) & (data_residuals['dni'] != 0)], building.building_info["name"])
    fig_dist.write_html(f"../../figures/pv_evaluation/{uuid}_distr.html")

    fig_true_vs_pred = plot_pred_vs_true(data_plot, building.building_info["name"])
    fig_true_vs_pred.write_html(f"../../figures/pv_evaluation/{uuid}_true_vs_pred.html")


if __name__ == "__main__":
    from src.building import load_anguillara

    anguillara = load_anguillara()
    for building in anguillara:
        if building.building_info["user_type"] != "consumer":
            evaluate_pv_model(building.building_info["id"])