import os
import torch
import pandas as pd
from pvlib.location import Location
from sklearn.model_selection import train_test_split

from settings import PROJECT_ROOT
from ..building import Building
from .data_handler import DataHandler
from .mlp import MultiLayerPerceptron
from .trainer import Trainer


def train(building: Building):
    """
    Funzione che addestra un modello di previsione della produzione fotovoltaica per un edificio. Il modello è un
    MultiLayerPerceptron con 2 hidden layers da 64 neuroni ciascuno. Il modello viene addestrato per massimo 300 epoche
    con early stopping se la loss validation non migliora di almeno 1e-6.

    Args:
        building (Building): oggetto Building con le informazioni sull'edificio.

    Returns:
        None
    """

    location = Location(latitude=building.building_info["coordinates"][1],
                        longitude=building.building_info["coordinates"][0])

    energy_data = building.energy_data.data
    weather_data = building.energy_data.weather_data

    data_handler = DataHandler(energy_data=energy_data, weather_data=weather_data)
    data = data_handler.create_data(location=location)

    X, y, x_scaler, y_scaler = data_handler.preprocess(data, building.building_info["id"])
    zero_mask = (X[:, 0] == 0) & (X[:, 1] == 0)
    X = X[~zero_mask]
    y = y[~zero_mask]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    train_loader = data_handler.create_dataloaders(X_train, y_train, batch_size=32)
    validation_loader = data_handler.create_dataloaders(X_val, y_val, batch_size=32)

    mlp = MultiLayerPerceptron(input_size=X.shape[1], hidden_layers=[64, 64], output_size=1)
    criterion = torch.nn.MSELoss()

    config = {
        'lr': 0.0001,
        'max_epochs': 300,
        'early_stopping_delta': 1e-6,
        'min_epochs': 50
    }

    trainer = Trainer(model=mlp, criterion=criterion, config=config)
    trainer.train(train_loader, validation_loader)

    torch.save(mlp.state_dict(),
               os.path.join(PROJECT_ROOT, "data", "anomaly_detection", "models", f"{building.building_info['id']}.pth"))
    trainer.evaluate(torch.tensor(X_val, dtype=torch.float32), y_val, y_scaler)

    df_loss = pd.DataFrame({"train": trainer.loss_list_train, "validation": trainer.loss_list_valid})

    df_loss.to_csv(os.path.join(PROJECT_ROOT, "data", "anomaly_detection", "loss", f"{building.building_info['id']}.csv"),
                   index=False)
