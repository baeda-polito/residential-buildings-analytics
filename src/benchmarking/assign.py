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
    cluster["cluster"] = cluster.apply(lambda row: 'C' + str(row['cluster']) if row['user_type'] == 'consumer' else 'P' + str(row['cluster']), axis=1)
    cluster["date"] = pd.to_datetime(cluster["date"]).dt.date

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

    # Separate data for consumers and prosumers
    df_consumer = df[df["user_type"] == "consumer"]
    df_prosumer = df[df["user_type"] != "consumer"]

    # For consumers
    centroids_consumer = df_consumer.groupby(["cluster", "hour"])["Load_norm"].mean().reset_index()
    centroids_consumer_pivot = centroids_consumer.pivot(index="cluster", columns="hour", values="Load_norm")

    df_consumer_pivot = df_consumer.pivot(index=["building_name", "date"], columns="hour", values="Load_norm")
    df_consumer_pivot.reset_index(inplace=True, level=[0, 1])
    df_consumer_pivot = pd.merge(df_consumer_pivot, cluster[["building_name", "date", "cluster"]], on=["building_name", "date"], how="inner")
    df_consumer_pivot = df_consumer_pivot.drop(columns=["building_name", "date"])

    # Find medoid, Q1, and Q3 for consumers
    medioid_consumer, q1_consumer, q3_consumer = find_medioid_and_quartiles(df_consumer_pivot, centroids_consumer_pivot)

    # For prosumers
    centroids_prosumer = df_prosumer.groupby(["cluster", "hour"])["Load_norm"].mean().reset_index()
    centroids_prosumer_pivot = centroids_prosumer.pivot(index="cluster", columns="hour", values="Load_norm")

    df_prosumer_pivot = df_prosumer.pivot(index=["building_name", "date"], columns="hour", values="Load_norm")
    df_prosumer_pivot.reset_index(inplace=True, level=[0, 1])
    df_prosumer_pivot = pd.merge(df_prosumer_pivot, cluster[["building_name", "date", "cluster"]], on=["building_name", "date"], how="inner")
    df_prosumer_pivot = df_prosumer_pivot.drop(columns=["building_name", "date"])

    # Find medoid, Q1, and Q3 for prosumers
    medioid_prosumer, q1_prosumer, q3_prosumer = find_medioid_and_quartiles(df_prosumer_pivot, centroids_prosumer_pivot)

    # Save results to CSV
    medioid_consumer.to_csv(f"../../results/medioid_{aggregate}_consumer.csv", index=True)
    q1_consumer.to_csv(f"../../results/q1_{aggregate}_consumer.csv", index=True)
    q3_consumer.to_csv(f"../../results/q3_{aggregate}_consumer.csv", index=True)

    medioid_prosumer.to_csv(f"../../results/medioid_{aggregate}_prosumer.csv", index=True)
    q1_prosumer.to_csv(f"../../results/q1_{aggregate}_prosumer.csv", index=True)
    q3_prosumer.to_csv(f"../../results/q3_{aggregate}_prosumer.csv", index=True)


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

        # Assign the cluster with the minimum distance
        clusters = data_pivot.apply(lambda row: medioids.apply(lambda x: euclidean(row, x), axis=1).idxmin(), axis=1)
        clusters = clusters.reset_index()
        clusters.columns = ["date", "cluster"]
        clusters["building_name"] = building.building_info["name"]
        clusters["user_type"] = building.building_info["user_type"]
        clusters["user_id"] = building.building_info["id"]

        building_data_list.append(clusters)

    df = pd.concat(building_data_list)

    df.to_csv(f"../../results/cluster_{aggregate}_assigned.csv", index=False)
