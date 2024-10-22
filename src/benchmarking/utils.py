import pandas as pd
from src.benchmarking import shape_factor
from scipy.spatial.distance import euclidean


def calculate_shape_factor(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i fattori di forma giornalieri per una serie temporale
    :param data: dataframe con la colonna "Load", contenente i dati di potenza elettrica consumata, e timestamp index
    :return: dataframe con i fattori di forma calcolati per ogni giorno disponibile
    """

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

    return df_sf


def find_medioid_and_quartiles(df_cluster, centroids_pivot):
    """
    Calcola i medioidi, il primo quartile (Q1) e il terzo quartile (Q3) dei profili in base alla distanza con il centroide del cluster.
    :param df_cluster: dataframe con i profili dei consumatori e prosumer
    :param centroids_pivot: dataframe con i centroidi dei cluster in pivot
    :return:
    """
    nearest_profiles = {}
    q1_profiles = {}
    q3_profiles = {}

    for cluster_label in centroids_pivot.index:
        # Get the centroid for the current cluster
        centroid = centroids_pivot.loc[cluster_label].values
        # Filter df_cluster by the current cluster
        cluster_profiles = df_cluster[df_cluster["cluster"] == cluster_label]
        # Drop the cluster column for distance calculation
        cluster_profiles_data = cluster_profiles.drop(columns=["cluster"])
        # Calculate distances for each profile in the current cluster
        distances = cluster_profiles_data.apply(lambda row: euclidean(row.values, centroid), axis=1)
        # Sort the profiles by distance
        sorted_profiles = distances.sort_values()

        # Find the medioid (the profile with the minimum distance)
        medioid_index = sorted_profiles.idxmin()
        nearest_profiles[cluster_label] = df_cluster.loc[medioid_index]

        # Find Q1 and Q3 based on sorted distances
        q1_index = sorted_profiles.index[int(len(sorted_profiles) * 0.25)]
        q3_index = sorted_profiles.index[int(len(sorted_profiles) * 0.75)]

        q1_profiles[cluster_label] = df_cluster.loc[q1_index]
        q3_profiles[cluster_label] = df_cluster.loc[q3_index]

    # Return medioid, Q1, and Q3 as DataFrames
    medioid_df = pd.DataFrame(nearest_profiles).T.drop(columns=["cluster"]).astype(float)
    q1_df = pd.DataFrame(q1_profiles).T.drop(columns=["cluster"]).astype(float)
    q3_df = pd.DataFrame(q3_profiles).T.drop(columns=["cluster"]).astype(float)

    return medioid_df, q1_df, q3_df

def create_dataset_classification():
    pass
