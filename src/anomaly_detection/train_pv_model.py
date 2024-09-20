import pandas as pd
import torch
from src.building import Building
from pvlib.location import Location
from src.anomaly_detection.data_handler import DataHandler
from sklearn.model_selection import train_test_split
from src.anomaly_detection.mlp import MultiLayerPerceptron
from src.anomaly_detection.trainer import Trainer


def train(uuid: str):
    building = Building(uuid=uuid, aggregate="anguillara")
    location = Location(latitude=building.building_info["coordinates"][1], longitude=building.building_info["coordinates"][0])

    energy_data = building.energy_meter.data
    weather_data = pd.read_csv("../../data/weather/anguillara.csv")

    data_handler = DataHandler(energy_data=energy_data, weather_data=weather_data)
    data = data_handler.create_data(location=location)

    X, y, x_scaler, y_scaler = data_handler.preprocess(data)
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
    torch.save(mlp.state_dict(), f"../../data/pv_add/models/{uuid}.pth")
    trainer.evaluate(torch.tensor(X_val, dtype=torch.float32), y_val, y_scaler)

    df_loss = pd.DataFrame({"train": trainer.loss_list_train, "validation": trainer.loss_list_valid})
    df_loss.to_csv(f"../../data/pv_add/loss/{uuid}.csv", index=False)


if __name__ == "__main__":
    from src.building import load_anguillara

    anguillara = load_anguillara()
    for building in anguillara[2:3]:
        if building.building_info["user_type"] != "consumer":
            print(f"Training model for {building.building_info['id']} --- {building.building_info['name']}")
            train(building.building_info["id"])