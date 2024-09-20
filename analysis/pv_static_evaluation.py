from src.anomaly_detection.mlp import MultiLayerPerceptron
from src.anomaly_detection.data_handler import DataHandler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from pvlib.location import Location
from src.building import Building, load_anguillara
import pandas as pd
import numpy as np
import torch
import json

anguillara = load_anguillara()
for building in anguillara:
    if building.building_info["user_type"] != "consumer":

        mlp = MultiLayerPerceptron(input_size=5, hidden_layers=[64, 64], output_size=1)
        mlp.load_state_dict(torch.load(f"../data/pv_add/models/{building.building_info['id']}.pth"))

        building = Building(uuid=building.building_info['id'], aggregate="anguillara")
        location = Location(latitude=building.building_info["coordinates"][1], longitude=building.building_info["coordinates"][0])

        energy_data = building.energy_meter.data
        weather_data = pd.read_csv("../data/weather/anguillara.csv")
        data_handler = DataHandler(energy_data=energy_data, weather_data=weather_data)
        data = data_handler.create_data(location=location)
        X, y, x_scaler, y_scaler = data_handler.preprocess(data)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

        y_pred_train = mlp(X_train_tensor).detach().numpy()

        y_pred_train_rescaled = y_scaler.inverse_transform(y_pred_train)
        y_true_train_rescaled = y_scaler.inverse_transform(y_train)

        y_pred_val = mlp(X_val_tensor).detach().numpy()

        y_pred_val_rescaled = y_scaler.inverse_transform(y_pred_val)
        y_true_val_rescaled = y_scaler.inverse_transform(y_val)

        # Subplots with 1 rows and two columns:
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        max_y_train = max(y_true_train_rescaled.max(), y_pred_train_rescaled.max())
        max_y_val = max(y_true_val_rescaled.max(), y_pred_val_rescaled.max())

        # Plot true vs pred for training data
        axs[0].scatter(y_true_train_rescaled, y_pred_train_rescaled, alpha=0.5)
        axs[0].plot([0, max_y_train], [0, max_y_train], color='red', linestyle='--')
        axs[0].set_title("Dataset train", fontsize=16)
        axs[0].set_xlabel("Valori reali [W]", fontsize=14)
        axs[0].set_ylabel("Valori predetti [W]", fontsize=14)

        # Plot true vs pred for validation data
        axs[1].scatter(y_true_val_rescaled, y_pred_val_rescaled, alpha=0.5)
        axs[1].plot([0, max_y_val], [0, max_y_val], color='red', linestyle='--')
        axs[1].set_title("Dataset di validazione", fontsize=16)
        axs[1].set_xlabel("Valori reali [W]", fontsize=14)
        axs[1].set_ylabel("Valori predetti [W]", fontsize=14)

        plt.suptitle("Predetto vs Reale", fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"../figures/pv_evaluation/{building.building_info['id']}_{building.building_info['name']}_true_vs_pred.png")

        # Plot residuals distributions
        residuals_train = y_pred_train_rescaled - y_true_train_rescaled
        residuals_val = y_pred_val_rescaled - y_true_val_rescaled

        max_train = max(abs(residuals_train))
        max_val = max(abs(residuals_val))

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        axs[0].hist(residuals_train, bins=50, color='green', alpha=0.7)
        axs[0].set_title("Dataset train", fontsize=16)
        axs[0].set_xlabel("Residui [W]", fontsize=14)
        axs[0].set_ylabel("Frequenza", fontsize=14)
        axs[0].set_xlim(-max_train, max_train)

        axs[1].hist(residuals_val, bins=50, color='green', alpha=0.7)
        axs[1].set_title("Dataset di validazione", fontsize=16)
        axs[1].set_xlabel("Residui [W]", fontsize=14)
        axs[1].set_ylabel("Frequenza", fontsize=14)
        axs[1].set_xlim(-max_val, max_val)

        plt.suptitle("Distribuzione dei residui", fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"../figures/pv_evaluation/{building.building_info['id']}_{building.building_info['name']}_distr.png")

        # Load loss data
        df_loss = pd.read_csv(f"../data/pv_add/loss/{building.building_info['id']}.csv")
        # Plot the loss function
        plt.figure(figsize=(10, 6))
        plt.plot(df_loss["train"], label="Train")
        plt.plot(df_loss["validation"], label="Validazione")
        plt.title("Andamento della funzione di costo", fontsize=18, fontweight='bold')
        plt.xlabel("Epoche", fontsize=14)
        plt.ylabel("Funzione di costo", fontsize=14)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=14, shadow=True, fancybox=True)
        plt.savefig(f"../figures/pv_evaluation/{building.building_info['id']}_{building.building_info['name']}_loss.png")

