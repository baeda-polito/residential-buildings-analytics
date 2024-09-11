import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pvlib.location import Location
from torch.utils.data import DataLoader, TensorDataset
import torch


class DataHandler:
    def __init__(self, energy_data, weather_data):
        self.energy_data = energy_data
        self.weather_data = weather_data

    def create_data(self, location):
        self.energy_data["timestamp"] = pd.to_datetime(self.energy_data["timestamp"])
        self.weather_data["timestamp"] = pd.to_datetime(self.weather_data["timestamp"])
        self.energy_data["timestamp"] = pd.to_datetime(self.energy_data["timestamp"])
        self.weather_data["timestamp"] = pd.to_datetime(self.weather_data["timestamp"])
        solar_angles = location.get_solarposition(times=self.weather_data["timestamp"]).reset_index()
        self.weather_data = pd.merge(self.weather_data, solar_angles[["timestamp", "azimuth", "zenith"]], on="timestamp",
                                     how="right")
        data_total = pd.merge(self.energy_data, self.weather_data, on="timestamp", how="left")
        data = data_total.dropna()
        data = data[["Production", "ghi", "dni", "air_temp", "azimuth", "zenith"]]

        return data

    @staticmethod
    def preprocess(data):
        X = data[["ghi", "dni", "air_temp", "azimuth", "zenith"]].to_numpy()
        y = data[["Production"]].to_numpy()

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_norm = x_scaler.fit_transform(X)
        y_norm = y_scaler.fit_transform(y)

        return X_norm, y_norm, x_scaler, y_scaler

    @staticmethod
    def create_dataloaders(X_train, y_train, batch_size):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_loader
