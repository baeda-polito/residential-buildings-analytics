import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from src.utils.pv_model import get_pv_production
from src.utils.pre_processing import (replace_constant_values, reconstruct_missing_values_interp,
                                      reconstruct_missing_values_knn)


def pre_process_power(data: pd.DataFrame, user_type: str, rated_power, rated_pv_power=None,
                      max_missing_interp: int = 4, max_missing_knn: int = 24):

    data_pre_processed = data[["timestamp", "power"]].copy()

    if user_type == "consumer":
        data_pre_processed.loc[data_pre_processed["power"] < 0, "power"] = np.nan
    else:
        if rated_pv_power is None:
            min_value = -10000
        else:
            min_value = -rated_pv_power * 1000
        data_pre_processed.loc[(data_pre_processed["power"] < min_value) | (data_pre_processed["power"] > -min_value), "power"] = np.nan

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

    # Check if in a day there are less than 12 values, save those indexes
    day_groups = data_model.groupby(data_model.index.date)

    # Not reconstruct if there are more than 6 hours of missing values
    index_not_reconstructable = []
    for day, group in day_groups:
        n_nan = group["productionPower"].isnull().sum()
        if n_nan > 4 * 6:
            index_nan = group[group["productionPower"].isnull()].index
            index_not_reconstructable.extend(index_nan)

    # Convert index not reconstructable into a series True/False with data_model index
    index_not_reconstructable = data_model.index.isin(index_not_reconstructable)
    index_not_reconstructable = pd.Series(index_not_reconstructable, index=data_model.index)
    data_model = data_model.drop(index_not_reconstructable.index)

    if physic_model:
        max_value = pv_params["rated_power"] * 1000 * 1.1
    else:
        max_value = 10000

    outliers = (data_model["productionPower"] > max_value) | (data_model["productionPower"] < 0)
    data_model.loc[outliers, "productionPower"] = np.nan
    index_not_phyisical = (data_model["productionPower"] < 10) & (data_model["ghi"] > 0)
    data_model.loc[index_not_phyisical, "productionPower"] = np.nan
    missing_values = data_model["productionPower"].isnull()
    data_to_reconstruct = data_model.loc[missing_values]
    data_to_reconstruct = data_to_reconstruct[~index_not_reconstructable]
    index_reconstruct = data_to_reconstruct.isin(index_not_reconstructable).index
    data_model = data_model.drop(index_reconstruct)

    if physic_model:
        pv_production = get_pv_production(
            lat=coordinates[0],
            lon=coordinates[1],
            tilt=pv_params["tilt"],
            azimuth=pv_params["azimuth"],
            rated_power=pv_params["rated_power"] * 1000,
            weather=weather_data)
        pv_production.index = pd.to_datetime(pv_production.index, utc=True)
        data_to_reconstruct.index = pd.to_datetime(data_to_reconstruct.index, utc=True)
        data_to_reconstruct.loc[data_to_reconstruct.index, "productionPower"] = pv_production.loc[data_to_reconstruct.index, "productionPower"]
        data_pre_processed = pd.concat([data_model, data_to_reconstruct])[['productionPower']]
        data_pre_processed.sort_index(inplace=True)
        data_pre_processed.reset_index(inplace=True)
        data_pre_processed = data_pre_processed[["timestamp", "productionPower"]]
    else:
        # Use data-driven model
        if len(data_to_reconstruct) < 0.7 * len(data_model):
            # TODO: Move into a function
            data_linear_model = data_model.dropna(subset=["productionPower"])
            data_linear_model.loc[data_model["ghi"] == 0, "productionPower"] = 0

            X_train = data_linear_model[["ghi", "dni"]]
            y_train = data_linear_model["productionPower"]

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Reconstruct the NaN with the model
            data_to_reconstruct["productionPower"] = model.predict(data_to_reconstruct[["ghi", "dni"]])
            data_to_reconstruct.loc[data_to_reconstruct["productionPower"] < 0, "productionPower"] = 0
            data_to_reconstruct.loc[data_to_reconstruct["ghi"] == 0, "productionPower"] = 0
            data_pre_processed = pd.concat([data_model, data_to_reconstruct])[['productionPower']]
            # Remove duplicates in index
            data_pre_processed = data_pre_processed[~data_pre_processed.index.duplicated(keep='first')]
            data_pre_processed.sort_index(inplace=True)
            data_pre_processed.reset_index(inplace=True)
            data_pre_processed = data_pre_processed[["timestamp", "productionPower"]]
        else:
            print("Too few data for reconstructing 'productionPower' in this period")
            data_pre_processed = pd.DataFrame(index=data_model.index, columns=["productionPower"])
            data_pre_processed["productionPower"] = np.nan
            data_pre_processed.reset_index(inplace=True, names=['timestamp'])

    return data_pre_processed

