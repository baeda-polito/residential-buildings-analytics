import pandas as pd
from src.building import load_anguillara, load_garda
from scipy.spatial.distance import euclidean


def calculate_medioids(aggregate: str):
    """

    Calcola i medioidi per i consumatori e i prosumer di un aggregato dopo aver applicato il clustering. I medioidi
    vengono salvati in due file csv.
    :param aggregate: nome dell'aggregato
    """

    cluster = pd.read_csv(f"../../results/cluster_{aggregate}.csv")
    cluster["cluster"] = cluster.apply(lambda row: 'C' + str(row['cluster']) if row['user_type'] == 'consumer' else 'P' + str(row['cluster']), axis=1)
    cluster["date"] = pd.to_datetime(cluster["date"]).dt.date

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

        cluster_user = cluster[cluster["building_name"] == building.building_info["name"]]
        data = pd.merge(data.drop(columns=["building_name"]), cluster_user, on="date", how="inner")
        building_data_list.append(data)

    df = pd.concat(building_data_list)

    df_consumer = df[df["user_type"] == "consumer"]
    df_prosumer = df[df["user_type"] != "consumer"]

    centroids_consumer = df_consumer.groupby(["cluster", "hour"])["Load_norm"].mean().reset_index()
    centroids_consumer_pivot = centroids_consumer.pivot(index="cluster", columns="hour", values="Load_norm")

    df_consumer_pivot = df_consumer.pivot(index=["building_name", "date"], columns="hour", values="Load_norm")
    df_consumer_pivot.reset_index(inplace=True, level=[0, 1])
    df_consumer_pivot = pd.merge(df_consumer_pivot, cluster[["building_name", "date", "cluster"]], on=["building_name", "date"], how="inner")
    df_consumer_pivot = df_consumer_pivot.drop(columns=["building_name", "date"])
    # For each cluster, find the nearest load profile in df_consumer_pivot
    nearest_profiles = {}
    for cluster_label in centroids_consumer_pivot.index:
        # Get the centroid for the current cluster
        centroid = centroids_consumer_pivot.loc[cluster_label].values
        # Filter df_consumer_pivot by the current cluster
        cluster_profiles = df_consumer_pivot[df_consumer_pivot["cluster"] == cluster_label]
        # Drop the cluster column for distance calculation
        cluster_profiles_data = cluster_profiles.drop(columns=["cluster"])
        # Calculate distances for each profile in the current cluster
        distances = cluster_profiles_data.apply(lambda row: euclidean(row.values, centroid), axis=1)
        # Find the index of the nearest profile
        nearest_index = distances.idxmin()
        # Store the nearest profile and its distance
        nearest_profiles[cluster_label] = df_consumer_pivot.loc[nearest_index]

    medioid_consumer = pd.DataFrame(nearest_profiles).T.drop(columns=["cluster"]).astype(float)
    medioid_consumer.to_csv(f"../../results/medioid_{aggregate}_consumer.csv", index=True)

    centroids_prosumer = df_prosumer.groupby(["cluster", "hour"])["Load_norm"].mean().reset_index()
    centroids_prosumer_pivot = centroids_prosumer.pivot(index="cluster", columns="hour", values="Load_norm")

    df_prosumer_pivot = df_prosumer.pivot(index=["building_name", "date"], columns="hour", values="Load_norm")
    df_prosumer_pivot.reset_index(inplace=True, level=[0, 1])
    df_prosumer_pivot = pd.merge(df_prosumer_pivot, cluster[["building_name", "date", "cluster"]], on=["building_name", "date"], how="inner")
    df_prosumer_pivot = df_prosumer_pivot.drop(columns=["building_name", "date"])

    # For each cluster, find the nearest load profile in df_prosumer_pivot
    nearest_profiles = {}
    for cluster_label in centroids_prosumer_pivot.index:
        # Get the centroid for the current cluster
        centroid = centroids_prosumer_pivot.loc[cluster_label].values
        # Filter df_prosumer_pivot by the current cluster
        cluster_profiles = df_prosumer_pivot[df_prosumer_pivot["cluster"] == cluster_label]
        # Drop the cluster column for distance calculation
        cluster_profiles_data = cluster_profiles.drop(columns=["cluster"])
        # Calculate distances for each profile in the current cluster
        distances = cluster_profiles_data.apply(lambda row: euclidean(row.values, centroid), axis=1)
        # Find the index of the nearest profile
        nearest_index = distances.idxmin()
        # Store the nearest profile and its distance
        nearest_profiles[cluster_label] = df_prosumer_pivot.loc[nearest_index]

    medioid_prosumer = pd.DataFrame(nearest_profiles).T.drop(columns=["cluster"]).astype(float)
    medioid_prosumer.to_csv(f"../../results/medioid_{aggregate}_prosumer.csv", index=True)


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
