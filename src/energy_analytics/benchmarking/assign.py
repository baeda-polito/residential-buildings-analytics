import os
import json
import pandas as pd
from loguru import logger
from scipy.spatial.distance import euclidean

from settings import PROJECT_ROOT
from ..aggregate import Aggregate


def calculate_medioids_profile(aggregate: Aggregate) -> None:
    """
    Calcola i profili di carico medioidi per i consumer e prosumer di un aggregato dopo aver applicato il clustering.
    Utilizza i medioidi calcolati con il metodo convenzionale (fattori di forma) e lo trasforma in profili di carico.
    Salve i profili medioidi in formato .csv nel folder 'results/benchmarking'.

    Args:
        aggregate (Aggregate): Oggetto Aggregate con la lista di edifici (List[Building]).

    Returns:
        None
    """

    if not os.path.exists(f"{PROJECT_ROOT}/results/benchmarking/cluster_{aggregate.name}.csv"):
        raise FileNotFoundError(f"File cluster_{aggregate.name}.csv not found in results/benchmarking. Run clustering first.")

    # Load cluster data
    cluster = pd.read_csv(f"{PROJECT_ROOT}/results/benchmarking/cluster_{aggregate.name}.csv")
    cluster["date"] = pd.to_datetime(cluster["date"]).dt.date

    # Load the medioids
    with open(f"{PROJECT_ROOT}/results/benchmarking/medioids_{aggregate.name}_consumer.json", "r") as f:
        medioids_consumer = json.load(f)

    with open(f"{PROJECT_ROOT}/results/benchmarking/medioids_{aggregate.name}_prosumer.json", "r") as f:
        medioids_prosumer = json.load(f)

    building_data_list = []
    for building in aggregate.buildings:
        data = building.energy_data.data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["hour"] = data["timestamp"].dt.strftime("%H:%M")
        data["date"] = data["timestamp"].dt.date
        data["building_name"] = building.building_info["name"]
        data["Load_norm"] = data["Load"] / data.groupby("date")["Load"].transform("max")

        cluster_user = cluster[cluster["building_name"] == building.building_info["name"]]
        data = pd.merge(data.drop(columns=["building_name"]), cluster_user, on="date", how="inner")
        building_data_list.append(data)

        # Concatenate all building data
    df = pd.concat(building_data_list)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    # Separate data for consumers and prosumers
    df_consumer = df[df["user_type"] == "consumer"]
    df_prosumer = df[df["user_type"] != "consumer"]

    logger.debug("Calcolo dei profili medioidi per CONSUMER e PROSUMER/PROSTORMER")

    medioids_profile_consumer = {}
    for cluster in medioids_consumer.keys():
        medioid_cluster = df_consumer[(df_consumer["building_name"] == medioids_consumer[cluster]["building_name"]) & (df_consumer["date"] == medioids_consumer[cluster]["date"])]
        medioid_cluster = medioid_cluster.pivot(index="date", columns="hour", values="Load_norm")
        medioid_cluster.index = [cluster]
        medioid_cluster = medioid_cluster.astype(float)

        medioids_profile_consumer[cluster] = medioid_cluster

    medioids_profile_consumer = pd.concat(medioids_profile_consumer, axis=0, ignore_index=True)
    medioids_profile_consumer.index = medioids_consumer.keys()
    medioids_profile_consumer.to_csv(os.path.join(PROJECT_ROOT, "results", "benchmarking", f"medioid_{aggregate.name}_consumer.csv"), index=True)

    medioids_profile_prosumer = {}
    for cluster in medioids_prosumer.keys():
        medioid_cluster = df_prosumer[(df_prosumer["building_name"] == medioids_prosumer[cluster]["building_name"]) & (df_prosumer["date"] == medioids_prosumer[cluster]["date"])]
        medioid_cluster = medioid_cluster.pivot(index="date", columns="hour", values="Load_norm")
        medioid_cluster.index = [cluster]
        medioid_cluster = medioid_cluster.astype(float)

        medioids_profile_prosumer[cluster] = medioid_cluster

    medioids_profile_prosumer = pd.concat(medioids_profile_prosumer, axis=0, ignore_index=True)
    medioids_profile_prosumer.index = medioids_prosumer.keys()
    medioids_profile_prosumer.to_csv(os.path.join(PROJECT_ROOT, "results", "benchmarking", f"medioid_{aggregate.name}_prosumer.csv"), index=True)


def assign_cluster(aggregate: Aggregate) -> None:
    """
    Assegna i cluster ai profili di carico degli edifici all'interno del cluster del medioide più vicino.
    Salva i profili di carico assegnati in formato .csv nel folder 'results/benchmarking'.

    Args:
        aggregate (Aggregate): Oggetto Aggregate con la lista di edifici (List[Building]).

    Returns:
        None
    """

    if not os.path.exists(f"{PROJECT_ROOT}/results/benchmarking/medioid_{aggregate.name}_consumer.csv"):
        raise FileNotFoundError(f"File medioid_{aggregate.name}_consumer.csv not found in results/benchmarking. Run calculate_medioids_profile first.")
    if not os.path.exists(f"{PROJECT_ROOT}/results/benchmarking/medioid_{aggregate.name}_prosumer.csv"):
        raise FileNotFoundError(f"File medioid_{aggregate.name}_prosumer.csv not found in results/benchmarking. Run calculate_medioids_profile first.")

    medioids_consumer = pd.read_csv(f"{PROJECT_ROOT}/results/benchmarking/medioid_{aggregate.name}_consumer.csv", index_col=0)
    medioids_prosumer = pd.read_csv(f"{PROJECT_ROOT}/results/benchmarking/medioid_{aggregate.name}_prosumer.csv", index_col=0)

    building_data_list = []
    for building in aggregate.buildings:
        data = building.energy_data.data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["hour"] = data["timestamp"].dt.strftime("%H:%M")
        data["date"] = data["timestamp"].dt.date
        data["building_name"] = building.building_info["name"]
        data["Load_norm"] = data["Load"] / data.groupby("date")["Load"].transform("max")
        data_pivot = data.pivot(index="date", columns="hour", values="Load_norm")

        if building.building_info["user_type"] == "consumer":
            medioids = medioids_consumer
        else:
            medioids = medioids_prosumer

        clusters = data_pivot.apply(
            lambda row: assign_to_nearest_or_anomalous(row, medioids), axis=1
        )
        clusters = clusters.reset_index()
        clusters.columns = ["date", "cluster"]
        clusters["building_name"] = building.building_info["name"]
        clusters["user_type"] = building.building_info["user_type"]
        clusters["user_id"] = building.building_info["id"]

        building_data_list.append(clusters)

    df = pd.concat(building_data_list)

    df.to_csv(os.path.join(PROJECT_ROOT, "results", "benchmarking", f"cluster_{aggregate.name}_assigned.csv"), index=False)


def assign_to_nearest_or_anomalous(load_profile: pd.Series, medioids: pd.DataFrame, threshold: float = 3.5):
    """
    Assegna un profilo di carico al cluster del medioide più vicino, a meno che la distanza non superi la soglia.
    Se la distanza è maggiore della soglia, il profilo viene assegnato al cluster "Anomalous".

    Args:
        load_profile (pd.Series): Profilo di carico da assegnare.
        medioids (pd.DataFrame): Profili medioidi dei cluster.
        threshold (float): Soglia massima di distanza per assegnare un profilo al cluster.

    Returns:
        str: Etichetta del cluster o "Anomalous".
    """
    # Calculate the Euclidean distance between the load profile and all medoids
    distances = medioids.apply(lambda x: euclidean(load_profile, x), axis=1)

    # Get the minimum distance and the corresponding cluster
    min_distance = distances.min()
    closest_cluster = distances.idxmin()

    # If the minimum distance exceeds the threshold, assign to "Anomalous"
    if min_distance > threshold:
        return "Anomalous"

    return closest_cluster
