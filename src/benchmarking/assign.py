import json
import pandas as pd
from src.building import load_anguillara, load_garda
from scipy.spatial.distance import euclidean
from src.benchmarking.utils import find_medioid_and_quartiles


def calculate_medioids(aggregate: str):
    """
    Calcola i medioidi, il primo quartile (Q1) e il terzo quartile (Q3) per i consumatori e i prosumer di un aggregato
    dopo aver applicato il clustering.
    :param aggregate: nome dell'aggregato
    """

    # Load cluster data
    cluster = pd.read_csv(f"../../results/cluster_{aggregate}.csv")
    cluster["date"] = pd.to_datetime(cluster["date"]).dt.date

    # Load the medioids
    with open(f"../../results/medioids_{aggregate}_consumer.json", "r") as f:
        medioids_consumer = json.load(f)

    with open(f"../../results/medioids_{aggregate}_prosumer.json", "r") as f:
        medioids_prosumer = json.load(f)

    building_list = []

    # Load appropriate building data
    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()

    # Collect building data
    building_data_list = []
    for building in building_list:
        data = building.energy_meter.data
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

    medioids_profile_consumer = {}
    for cluster in medioids_consumer.keys():
        medioid_cluster = df_consumer[(df_consumer["building_name"] == medioids_consumer[cluster]["building_name"]) & (df_consumer["date"] == medioids_consumer[cluster]["date"])]
        medioid_cluster = medioid_cluster.pivot(index="date", columns="hour", values="Load_norm")
        medioid_cluster.index = [cluster]
        medioid_cluster = medioid_cluster.astype(float)

        medioids_profile_consumer[cluster] = medioid_cluster

    medioids_profile_consumer = pd.concat(medioids_profile_consumer, axis=0, ignore_index=True)
    medioids_profile_consumer.index = medioids_consumer.keys()
    medioids_profile_consumer.to_csv(f"../../results/medioid_{aggregate}_consumer.csv", index=True)

    medioids_profile_prosumer = {}
    for cluster in medioids_prosumer.keys():
        medioid_cluster = df_prosumer[(df_prosumer["building_name"] == medioids_prosumer[cluster]["building_name"]) & (df_prosumer["date"] == medioids_prosumer[cluster]["date"])]
        medioid_cluster = medioid_cluster.pivot(index="date", columns="hour", values="Load_norm")
        medioid_cluster.index = [cluster]
        medioid_cluster = medioid_cluster.astype(float)

        medioids_profile_prosumer[cluster] = medioid_cluster

    medioids_profile_prosumer = pd.concat(medioids_profile_prosumer, axis=0, ignore_index=True)
    medioids_profile_prosumer.index = medioids_prosumer.keys()
    medioids_profile_prosumer.to_csv(f"../../results/medioid_{aggregate}_prosumer.csv", index=True)


def assign_cluster(aggregate: str):
    """
    Riassegna ogni profile di carico al medioide più vicino
    :param aggregate: nome dell'aggregato (anguillara o garda)
    """

    medioids_consumer = pd.read_csv(f"../../results/medioid_{aggregate}_consumer.csv", index_col=0)
    medioids_prosumer = pd.read_csv(f"../../results/medioid_{aggregate}_prosumer.csv", index_col=0)

    building_list = []
    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()

    building_data_list = []
    for building in building_list:
        data = building.energy_meter.data
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

    df.to_csv(f"../../results/cluster_{aggregate}_assigned.csv", index=False)


def assign_to_nearest_or_anomalous(load_profile, medioids, threshold=3):
    """
    Assegna l'etichetta di cluster al profilo di carico più vicino, a meno che la distanza non superi la soglia.
    Se la distanza è maggiore della soglia, il profilo viene assegnato ad 'Anomalous'.
    :param load_profile: profilo di carico da assegnare
    :param medioids: medioidi dei cluster
    :param threshold: soglia di distanza
    :return: etichetta del cluster  o 'Anomalous'
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


calculate_medioids("anguillara")
assign_cluster("anguillara")