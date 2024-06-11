from building import load_anguillara, load_garda
from settings import PROJECT_ROOT
import shape_factor
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

global building_list

if __name__ == "__main__":
    aggregate = "garda"

    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()

    building_sf_dict = {}

    for building in building_list:
        data = building.energy_meter.data.copy()
        data.set_index("timestamp", inplace=True)
        data = data["Net"].clip(lower=0)
        data = data.resample("1H").mean()
        mean_max = shape_factor.sf_mean_max(data)
        min_max = shape_factor.sf_min_max(data)
        min_mean = shape_factor.sf_min_mean(data)
        daytime_mean = shape_factor.sf_daytime_mean(data)
        daytime_max = shape_factor.sf_daytime_max(data)
        daytime_min_mean = shape_factor.sf_daytime_min_mean(data)
        nightime_mean = shape_factor.sf_nightime_mean(data)
        nightime_max = shape_factor.sf_nightime_max(data)
        nightime_min_mean = shape_factor.sf_nightime_min_mean(data)
        afternoon_mean = shape_factor.sf_afternoon_mean(data)
        afternoon_max = shape_factor.sf_afternoon_max(data)
        afternoon_min_mean = shape_factor.sf_afternoon_min_mean(data)
        peak_load = shape_factor.peakload(data)
        peak_period = shape_factor.peak_period(data, peak_load)

        df_sf = pd.concat([mean_max, min_max, min_mean, daytime_mean, daytime_max, daytime_min_mean, nightime_mean,
                           nightime_max, nightime_min_mean, afternoon_mean, afternoon_max, afternoon_min_mean,
                           peak_period], axis=1)
        df_sf.columns = ["mean_max", "min_max", "min_mean", "daytime_mean", "daytime_max", "daytime_min_mean",
                         "nightime_mean", "nightime_max", "nightime_min_mean", "afternoon_mean", "afternoon_max",
                         "afternoon_min_mean", "peak_night", "pick_morning", "peak_mid-day", "peak_evening"]
        df_sf.dropna(inplace=True)
        df_sf["building_name"] = building.building_info["name"]
        df_sf["user_type"] = building.building_info["user_type"]
        building_sf_dict[building.building_info["name"]] = df_sf

    df_sf_total = pd.concat(building_sf_dict.values())

    """
    Consumer clustering
    """

    df_sf_total_norm = df_sf_total.copy()
    df_sf_total_norm = df_sf_total_norm[df_sf_total_norm["user_type"] == "consumer"]
    df_sf_total_reset = df_sf_total_norm.reset_index()
    df_sf_total_reset.iloc[:, 1:14] = (df_sf_total_reset.iloc[:, 1:14] - df_sf_total_reset.iloc[:, 1:14].min()) / (
                df_sf_total_reset.iloc[:, 1:14].max() - df_sf_total_reset.iloc[:, 1:14].min())
    df_sf_total_norm = df_sf_total_reset.set_index("index")

    silhouette_scores = []
    for n_clusters in range(2, 6):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(df_sf_total_norm.iloc[:, :-2])
        silhouette_scores.append(silhouette_score(df_sf_total_norm.iloc[:, :-2], kmeans.labels_))

    n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(df_sf_total_norm.iloc[:, :-2])
    df_sf_total_norm["cluster"] = kmeans.labels_

    df_sf_consumer = df_sf_total_norm.copy()

    """
    Clustering producer/prostormer
    """

    df_sf_total_norm = df_sf_total.copy()
    df_sf_total_norm = df_sf_total_norm[df_sf_total_norm["user_type"] != "consumer"]
    df_sf_total_reset = df_sf_total_norm.reset_index()
    df_sf_total_reset.iloc[:, 1:14] = (df_sf_total_reset.iloc[:, 1:14] - df_sf_total_reset.iloc[:, 1:14].min()) / (
                df_sf_total_reset.iloc[:, 1:14].max() - df_sf_total_reset.iloc[:, 1:14].min())
    df_sf_total_norm = df_sf_total_reset.set_index("index")

    silhouette_scores = []
    for n_clusters in range(2, 6):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(df_sf_total_norm.iloc[:, :-2])
        silhouette_scores.append(silhouette_score(df_sf_total_norm.iloc[:, :-2], kmeans.labels_))

    n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(df_sf_total_norm.iloc[:, :-2])
    df_sf_total_norm["cluster"] = kmeans.labels_

    df_sf_producer = df_sf_total_norm.copy()

    df_sf_total = pd.concat([df_sf_consumer, df_sf_producer])
    df_sf_total.reset_index(inplace=True, names=["date"])

    df_sf_total[['date', 'building_name', "user_type", 'cluster']].to_csv(
        os.path.join(PROJECT_ROOT, "results", f"cluster_{aggregate}.csv"),
        index=False)
