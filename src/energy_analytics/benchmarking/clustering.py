import os
import json
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.neighbors import LocalOutlierFactor

from settings import PROJECT_ROOT
from ..aggregate import Aggregate
from .utils import calculate_shape_factor
from . import viz


def run_clustering(aggregate: Aggregate):
    """
    Esegue il clustering per consumer e producer/prosumer utilizzando i fattori di forma. Salva un file .csv con i
    risultati del clustering nel folder 'results/benchmarking'. Salva i medioidi dei cluster in formato json (data e edificio)
    e l'assegnazione dei cluster con il metodo convenzionale (no online). La riassegnazione online viene fatta tramite
    il modulo 'assign'.

    Args:
        aggregate (Aggregate): Oggetto Aggregate con la lista di edifici (List[Building]).

    Returns:
        None
    """

    building_sf_dict = {}

    for building in aggregate.buildings:
        # Calculating the shape factors for all building and all days available
        data = building.energy_data.data.copy()
        data.set_index("timestamp", inplace=True)
        data = data["Load"].clip(lower=0)
        data = data.resample("1H").mean()
        df_sf = calculate_shape_factor(data)
        df_sf["building_name"] = building.building_info["name"]
        df_sf["user_type"] = building.building_info["user_type"]
        df_sf["user_id"] = building.building_info["id"]
        building_sf_dict[building.building_info["name"]] = df_sf

    df_sf_total = pd.concat(building_sf_dict.values())

    cluster_range = range(2, 5)

    """
    Consumer clustering
    """
    logger.debug("Procedura di clustering per gli utenti CONSUMER")

    df_sf_total_norm = df_sf_total.copy()
    df_sf_total_norm = df_sf_total_norm[df_sf_total_norm["user_type"] == "consumer"]
    df_sf_total_reset = df_sf_total_norm.reset_index()

    # Outlier detection with LOF
    lof = LocalOutlierFactor(n_neighbors=50)
    outliers = lof.fit_predict(df_sf_total_reset.set_index("index").drop(columns=["building_name", "user_type", "user_id"]))
    outliers_index = df_sf_total_reset.index[outliers == -1].tolist()
    df_sf_total_reset = df_sf_total_reset.drop(outliers_index)

    # Min-max normalization
    df_sf_total_reset.iloc[:, 1:12] = (df_sf_total_reset.iloc[:, 1:12] - df_sf_total_reset.iloc[:, 1:12].min()) / (
                df_sf_total_reset.iloc[:, 1:12].max() - df_sf_total_reset.iloc[:, 1:12].min())
    df_sf_total_norm = df_sf_total_reset.set_index("index")

    silhouette_scores = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(df_sf_total_norm.iloc[:, :-3])
        silhouette_scores.append(silhouette_score(df_sf_total_norm.iloc[:, :-3], kmeans.labels_))

    n_clusters = silhouette_scores.index(max(silhouette_scores)) + cluster_range[0]

    df_score_consumer = pd.DataFrame(
        {"n_clusters": cluster_range, "silhouette_score": silhouette_scores}
    )

    fig_score = viz.plot_silhouette_scores(df_score_consumer)
    fig_score.savefig(os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"silhouette_score_consumer_{aggregate.name}.png"))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(df_sf_total_norm.iloc[:, :-3])
    df_sf_total_norm["cluster"] = kmeans.labels_
    df_sf_total_norm["cluster"] = df_sf_total_norm["cluster"].apply(lambda x: f"C{x}")

    # Calculate the medioid of each cluster
    medioids_consumer = {}
    for cluster in df_sf_total_norm["cluster"].unique():
        data_cluster = df_sf_total_norm[df_sf_total_norm["cluster"] == cluster].drop(columns=["building_name", "user_type", "user_id", "cluster"])
        distances = pairwise_distances(data_cluster, metric="euclidean")

        row_sums = distances.sum(axis=1)
        medioid_index = row_sums.argmin()

        medioid = df_sf_total_norm[df_sf_total_norm["cluster"] == cluster].reset_index(names="date").iloc[medioid_index]
        medioid = medioid[["date", "building_name"]]
        medioid["date"] = medioid["date"].strftime("%Y-%m-%d")

        medioids_consumer[cluster] = medioid.to_dict()

    df_sf_consumer = df_sf_total_norm.copy()

    """
    Clustering producer/prostormer
    """
    logger.debug("Procedura di clustering per gli utenti PROSUMER/PROSTORMER")

    df_sf_total_norm = df_sf_total.copy()
    df_sf_total_norm = df_sf_total_norm[df_sf_total_norm["user_type"] != "consumer"]
    df_sf_total_reset = df_sf_total_norm.reset_index()

    lof = LocalOutlierFactor(n_neighbors=50)
    outliers = lof.fit_predict(
        df_sf_total_reset.set_index("index").drop(columns=["building_name", "user_type", "user_id"]))
    outliers_index = df_sf_total_reset.index[outliers == -1].tolist()
    df_sf_total_reset = df_sf_total_reset.drop(outliers_index)

    df_sf_total_reset.iloc[:, 1:12] = (df_sf_total_reset.iloc[:, 1:12] - df_sf_total_reset.iloc[:, 1:12].min()) / (
                df_sf_total_reset.iloc[:, 1:12].max() - df_sf_total_reset.iloc[:, 1:12].min())
    df_sf_total_norm = df_sf_total_reset.set_index("index")

    silhouette_scores = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(df_sf_total_norm.iloc[:, :-3])
        silhouette_scores.append(silhouette_score(df_sf_total_norm.iloc[:, :-3], kmeans.labels_))

    n_clusters = silhouette_scores.index(max(silhouette_scores)) + cluster_range[0]

    df_score_prosumer = pd.DataFrame(
        {"n_clusters": cluster_range, "silhouette_score": silhouette_scores}
    )

    fig_score = viz.plot_silhouette_scores(df_score_prosumer)
    fig_score.savefig(os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"silhouette_score_prosumer_{aggregate.name}.png"))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(df_sf_total_norm.iloc[:, :-3])
    df_sf_total_norm["cluster"] = kmeans.labels_
    df_sf_total_norm["cluster"] = df_sf_total_norm["cluster"].apply(lambda x: f"P{x}")

    df_sf_producer = df_sf_total_norm.copy()

    # Calculate the medioid of each cluster
    medioids_prosumer = {}
    for cluster in df_sf_total_norm["cluster"].unique():
        data_cluster = df_sf_total_norm[df_sf_total_norm["cluster"] == cluster].drop(columns=["building_name", "user_type", "user_id", "cluster"])
        distances = pairwise_distances(data_cluster, metric="euclidean")

        row_sums = distances.sum(axis=1)
        medioid_index = row_sums.argmin()

        medioid = df_sf_total_norm[df_sf_total_norm["cluster"] == cluster].reset_index(names="date").iloc[medioid_index]
        medioid = medioid[["date", "building_name"]]
        medioid["date"] = medioid["date"].strftime("%Y-%m-%d")

        medioids_prosumer[cluster] = medioid.to_dict()

    df_sf_total = pd.concat([df_sf_consumer, df_sf_producer])
    df_sf_total.reset_index(inplace=True, names=["date"])

    df_sf_total[['date', 'building_name', "user_id", "user_type", 'cluster']].to_csv(
        os.path.join(PROJECT_ROOT, "results", "benchmarking", f"cluster_{aggregate.name}.csv"),
        index=False)

    # Save medioids in json
    with open(os.path.join(PROJECT_ROOT, "results", "benchmarking",  f"medioids_{aggregate.name}_consumer.json"), "w") as f:
        json.dump(medioids_consumer, f)

    with open(os.path.join(PROJECT_ROOT, "results", "benchmarking", f"medioids_{aggregate.name}_prosumer.json"), "w") as f:
        json.dump(medioids_prosumer, f)
