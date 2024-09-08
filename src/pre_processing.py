import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from settings import PROJECT_ROOT
from src.utils.pv_model import get_pv_production
from src.utils.pre_processing import (replace_constant_values, reconstruct_missing_values_interp,
                                      reconstruct_missing_values_knn)


def pre_process_power(data: pd.DataFrame, user_type: str, rated_power, rated_pv_power=None,
                      max_missing_interp: int = 4, max_missing_knn: int = 24):

    data_pre_processed = data[["timestamp", "power"]].copy()

    if user_type == "consumer":
        data_pre_processed.loc[data_pre_processed["power"] < 0, "power"] = np.nan
    else:
        data_pre_processed.loc[(data_pre_processed["power"] < -rated_pv_power * 1000) | (data_pre_processed["power"] > rated_power * 1000), "power"] = np.nan

    data_pre_processed.loc[:, "power"] = replace_constant_values(data["power"], 4)

    data_pre_processed_lin = reconstruct_missing_values_interp(data_pre_processed[["timestamp", "power"]].copy(),
                                                               max_missing=max_missing_interp)

    data_pre_processed_knn = reconstruct_missing_values_knn(data_pre_processed_lin.copy(),
                                                            k=5,
                                                            min_missing=max_missing_interp + 1,
                                                            max_missing=max_missing_knn)

    data_pre_processed = data_pre_processed_knn.copy()

    return data_pre_processed[["timestamp", "power"]]


def pre_process_production_power(data: pd.DataFrame, weather_data: pd.DataFrame,
                                 physic_model: bool = False, pv_params: dict = None, coordinates: list = None):

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])

    data_model = pd.merge(data, weather_data, on="timestamp", how="right")
    data_model.set_index("timestamp", inplace=True)

    missing_values = data_model["productionPower"].isnull()
    outliers = (data_model["productionPower"] > pv_params["rated_power"] * 1000) | (data_model["productionPower"] < 0)
    index_not_phyisical = data_model["productionPower"] < - data_model["power"]
    data_to_reconstruct = data_model.loc[missing_values | index_not_phyisical | outliers]

    if physic_model:
        data_pre_processed = get_pv_production(
            lat=coordinates[1],
            lon=coordinates[0],
            tilt=pv_params["tilt"],
            azimuth=pv_params["azimuth"],
            rated_power=pv_params["rated_power"] * 1000,
            weather=weather_data)
        data_pre_processed.reset_index(inplace=True)
    else:
        # Use data-driven model
        if len(data_to_reconstruct) < 0.7 * len(data_model):
            # TODO: Move into a function
            data_model.dropna(subset=["productionPower"], inplace=True)
            data_model = data_model[~index_not_phyisical]
            data_model = data_model[~outliers]
            data_model.loc[data_model["ghi"] == 0, "productionPower"] = 0

            X_train = data_model[["ghi", "dni"]]
            y_train = data_model["productionPower"]

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Reconstruct the NaN with the model
            data_to_reconstruct["productionPower"] = model.predict(data_to_reconstruct[["ghi", "dni"]])
            data_to_reconstruct.loc[data_to_reconstruct["productionPower"] < 0, "productionPower"] = 0
            data_to_reconstruct.loc[data_to_reconstruct["ghi"] == 0, "productionPower"] = 0
            data_pre_processed = pd.concat([data_model, data_to_reconstruct])[['productionPower']]
            data_pre_processed.sort_index(inplace=True)
            data_pre_processed.reset_index(inplace=True)
            data_pre_processed = data_pre_processed[["timestamp", "productionPower"]]
        else:
            print("Too few data for reconstructing 'productionPower' in this period")
            data_pre_processed = pd.DataFrame(index=data_model.index, columns=["productionPower"])
            data_pre_processed["productionPower"] = np.nan
            data_pre_processed.reset_index(inplace=True, names=['timestamp'])

    return data_pre_processed


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "7436df46-294b-4c97-bd1b-8aaa3aed97c5.csv"))
    weather = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "weather", "anguillara.csv"))
    # Delete the "+00:00" from the timestamp
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather["timestamp"] = weather["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    power = pre_process_power(df, "prostormer", 6, 6)
    power["timestamp"] = pd.to_datetime(power["timestamp"])
    power.set_index("timestamp", inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df["power"] = power["power"]
    df.reset_index(inplace=True)

    production_power = pre_process_production_power(df, weather, 6, physic_model=True,
                                                    pv_params={"lat": 42.0837, "lon": 12.283, "tilt": 30, "azimuth": 0, "rated_power": 6000})
    production_power.set_index("timestamp", inplace=True)

    df.set_index("timestamp", inplace=True)
    df.loc[production_power.index, "productionPower"] = production_power["productionPower"]
