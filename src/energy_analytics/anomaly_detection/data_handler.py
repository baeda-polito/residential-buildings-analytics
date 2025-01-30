import os
import torch
import pvlib
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from settings import PROJECT_ROOT


class DataHandler:
    """
    Classe per gestire i dati relativi alla produzione e al meteo. I dati vengono integrati in un unico dataframe,
    possono essere scalati tra 0 e 1 e infine divisi in batch, creando dei dataloaders, per l'addestramento del modello.

    Args:
        energy_data (pd.DataFrame): dataframe con i dati relativi alla produzione. Deve contenere una colonna "timestamp" con le date
        e la produzione in una colonna "Production".
        weather_data (pd.DataFrame): dataframe con i dati relativi al meteo (ghi, dni, air_temp, azimuth, zenith). Deve contenere una
        colonna "timestamp" con le date.

    """
    def __init__(self, energy_data, weather_data):
        self.energy_data = energy_data
        self.weather_data = weather_data

    def create_data(self, location: pvlib.location.Location):
        """
        Crea un dataframe unico con i dati relativi alla produzione e al meteo. Aggiunge le informazioni sugli angoli solari
        e rimuove le righe con valori mancanti.

        Args:
            location (pvlib.location.Location): oggetto con le informazioni sulla posizione geografica.

        Returns:
            pd.DataFrame: dataframe con i dati relativi alla produzione e al meteo contenente le colonne "Production", "ghi", "dni", "air_temp", "azimuth", "zenith".
        """
        self.energy_data["timestamp"] = pd.to_datetime(self.energy_data["timestamp"])
        self.weather_data["timestamp"] = pd.to_datetime(self.weather_data["timestamp"]).dt.tz_localize(None)
        solar_angles = location.get_solarposition(times=self.weather_data["timestamp"]).reset_index()
        self.weather_data = pd.merge(self.weather_data, solar_angles[["timestamp", "azimuth", "zenith"]], on="timestamp",
                                     how="right")
        data_total = pd.merge(self.energy_data, self.weather_data, on="timestamp", how="left")
        data = data_total.dropna()
        data.set_index("timestamp", inplace=True)
        data = data[["Production", "ghi", "dni", "air_temp", "azimuth", "zenith"]]

        return data

    @staticmethod
    def preprocess(data: pd.DataFrame, uuid: str, save_scalers=True):
        """
        Preprocessa i dati scalando le variabili tra 0 e 1.

        Args:
            data (pd.DataFrame): dataframe con i dati da preprocessare.
            uuid (str): identificativo dell'edificio.
            save_scalers (bool, optional): se True salva i MinMaxScaler. Default True.

        Returns:
            tuple: tuple contente X_norm, y_norm, x_scaler, y_scaler.
        """
        X = data[["ghi", "dni", "air_temp", "azimuth", "zenith"]].to_numpy()
        y = data[["Production"]].to_numpy()

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_norm = x_scaler.fit_transform(X)
        y_norm = y_scaler.fit_transform(y)

        if save_scalers:

            with open(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "scalers", f"x_scaler_{uuid}.pkl"), "wb") as f:
                pickle.dump(x_scaler, f)

            with open(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "scalers", f"y_scaler_{uuid}.pkl"), "wb") as f:
                pickle.dump(y_scaler, f)

        return X_norm, y_norm, x_scaler, y_scaler

    @staticmethod
    def create_dataloaders(X_train: np.array, y_train: np.array, batch_size: int):
        """
        Crea un DataLoader.

        Args:
            X_train (np.array): array con le variabili indipendenti.
            y_train (np.array): array con la variabile dipendente.
            batch_size (int): dimensione del batch.

        Returns:
            DataLoader: DataLoader per l'addestramento del modello.
        """

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_loader
