import pandas as pd
from src.building import load_anguillara, load_garda
from src.benchmarking.utils import calculate_shape_factor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
import os
from settings import PROJECT_ROOT
import joblib


def run_user_classification(aggregate: str):
    """
    Allena i modelli di classificazione del tipo CART per stimare la probabilità di un nuovo utente di appartenere a
    uno dei cluster identificati.
    :param aggregate: il nome dell'aggregato
    :return: None
    """

    building_list = []

    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()

    building_info_list = []
    shape_factor_list = []
    for building in building_list:
        building_info = {
            "building_name": building.building_info["name"],
            "user_type": building.building_info["user_type"],
            "persons": building.building_info["persons"],
            "occupancy_night": building.building_info["occupancy"]["24-8"] / building.building_info["persons"],
            "occupancy_morning": building.building_info["occupancy"]["8-13"] / building.building_info["persons"],
            "occupancy_afternoon": building.building_info["occupancy"]["13-19"] / building.building_info["persons"],
            "occupancy_evening": building.building_info["occupancy"]["19-24"] / building.building_info["persons"],
            "rated_power": building.building_info["rated_power"],
            "ac": 1 if building.building_info["ac"]["n_ac"] > 0 else 0,
            "pv_power": building.building_info["pv"]["rated_power"],
            "storage": 1 if building.building_info["pv"]["storage"] else 0
        }
        building_info_list.append(building_info)

        # Calculate shape factor
        data = building.energy_meter.data.copy()
        data.set_index("timestamp", inplace=True)
        data = data["Load"].clip(lower=0)
        data = data.resample("1H").mean()
        building_sf = calculate_shape_factor(data)
        building_sf["building_name"] = building.building_info["name"]
        building_sf["date"] = building_sf.index
        building_sf["user_type"] = building.building_info["user_type"]
        shape_factor_list.append(building_sf)

    df_info = pd.DataFrame(building_info_list)
    df_info.fillna(0, inplace=True)

    df_sf = pd.concat(shape_factor_list)
    df_sf.reset_index(inplace=True, drop=True)

    """
    Outlier detection
    """
    lof = LocalOutlierFactor(n_neighbors=50)
    outliers = lof.fit_predict(df_sf.drop(columns=["building_name", "date", "user_type"]))

    outliers_index = df_sf[outliers == -1].index
    outliers_record = df_sf.loc[outliers_index][["building_name", "date"]].reset_index(drop=True)
    outliers_record["date"] = pd.to_datetime(outliers_record["date"]).dt.strftime("%Y-%m-%d")

    cluster = pd.read_csv(f"../../results/cluster_{aggregate}_assigned.csv")
    # Drop "Anomalous" from "cluster" column
    cluster = cluster[cluster["cluster"] != "Anomalous"]

    # merge df_info and cluster on "name"
    df = pd.merge(df_info, cluster.drop(columns=["user_id", "user_type"]), on="building_name")
    # Drop the outliers_record from df using both "building_name" and "date"
    df = df[
        ~df.set_index(["building_name", "date"]).index.isin(outliers_record.set_index(["building_name", "date"]).index)]

    """
    Feature engineering
    """

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["day_type"] = df["date"].apply(
        lambda x: "sunday" if x.weekday() == 6 else "saturday" if x.weekday() == 5 else "weekday")
    df["day_type"] = df["day_type"].astype("category").cat.codes
    df["month"] = pd.to_datetime(df["date"]).dt.month

    df_consumer = df[df["user_type"] == "consumer"].drop(
        columns=["building_name", "user_type", "date", "pv_power", "storage"])
    df_prosumer = df[df["user_type"] != "consumer"].drop(columns=["building_name", "user_type", "date"])

    """
    Consumer classification
    """

    df_consumer.sort_values(by="cluster", inplace=True)

    X_consumer = df_consumer.drop(columns=["cluster"]).to_numpy()
    y_consumer = pd.Categorical(df_consumer["cluster"]).codes

    model_consumer = DecisionTreeClassifier(min_samples_leaf=10, max_depth=7, random_state=42)
    model_consumer.fit(X_consumer, y_consumer)

    # Save the model
    joblib.dump(model_consumer,
                os.path.join(PROJECT_ROOT, "data", "benchmarking", "models", f"model_consumer{aggregate}.pkl"))

    """
    Prosumer classification
    """

    df_prosumer.sort_values(by="cluster", inplace=True)

    X_prosumer = df_prosumer.drop(columns=["cluster"]).to_numpy()
    y_prosumer = pd.Categorical(df_prosumer["cluster"]).codes

    model_prosumer = DecisionTreeClassifier(min_samples_leaf=10, max_depth=7, random_state=42)
    model_prosumer.fit(X_prosumer, y_prosumer)

    # Save the model
    joblib.dump(model_prosumer,
                os.path.join(PROJECT_ROOT, "data", "benchmarking", "models", f"model_prosumer_{aggregate}.pkl"))
