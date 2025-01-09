import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..building import Building
from ..aggregate import Aggregate
from .utils import calculate_shape_factor
from settings import PROJECT_ROOT


def plot_silhouette_scores(scores: pd.DataFrame) -> plt.Figure:
    """
    Grafica il Silhouette score per ogni cluster.

    Args:
        scores (pd.DataFrame): DataFrame con i Silhouette scores per ogni cluster.

    Returns:
        plt.Figure: Figure di matplotlib.
    """

    max_score = scores["silhouette_score"].max()
    max_cluster = scores[scores["silhouette_score"] == max_score]["n_clusters"].values[0]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(scores["n_clusters"], scores["silhouette_score"])
    plt.scatter(max_cluster, max_score, color='red')
    plt.hlines(max_score, scores["n_clusters"].min(), scores["n_clusters"].max(), linestyles='dashed', colors='red')
    plt.vlines(max_cluster, scores["silhouette_score"].min(), scores["silhouette_score"].max(), linestyles='dashed',
               colors='red')
    plt.xlabel("Numero di cluster")
    plt.ylabel("Silhouette score")
    plt.xticks(scores["n_clusters"])
    plt.title("Silhouette score al variare del numero di cluster")

    return fig


def plot_load_profiles_user(building: Building, cluster: pd.DataFrame) -> None:
    """
    Grafica i profili di carico di un utente dividendoli per cluster. Salva la figura in 'figures/benchmarking'.

    Args:
        building (Building): Oggetto Building.
        cluster (pd.DataFrame): DataFrame con i cluster assegnati agli utenti.

    Returns:
        None
    """

    data = building.energy_data.data

    cluster_user = cluster[cluster["user_id"] == building.building_info["id"]]
    cluster_user["date"] = pd.to_datetime(cluster_user["date"]).dt.date

    # Merge data with cluster user on the date
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date

    data_cluster = pd.merge(data[["timestamp", "date", "Load"]], cluster_user[["date", "cluster"]], on="date",
                            how="inner")
    data_cluster["hour"] = data_cluster["timestamp"].dt.strftime("%H:%M")

    cluster_labels = cluster.loc[cluster["user_type"] == building.building_info["user_type"], "cluster"].unique()
    # Drop the 'Anomalous' if present
    if "Anomalous" in cluster_labels:
        cluster_labels = cluster_labels[cluster_labels != "Anomalous"]
    cluster_labels.sort()

    centroids = data_cluster.groupby(["hour", "cluster"])["Load"].mean().reset_index()
    q1 = data_cluster.groupby(["hour", "cluster"])["Load"].quantile(0.25).reset_index()
    q3 = data_cluster.groupby(["hour", "cluster"])["Load"].quantile(0.75).reset_index()

    fig, ax = plt.subplots(figsize=(14, 4), nrows=1, ncols=(cluster_labels.size + 1), sharey=True)
    palette = sns.color_palette("colorblind", cluster_labels.size)

    # Adding the profiles
    for i, cluster_label in enumerate(cluster_labels):
        data_cluster_viz = data_cluster[data_cluster["cluster"] == cluster_label]
        data_cluster_viz_grouped = data_cluster_viz.groupby("date")

        for date, group in data_cluster_viz_grouped:
            ax[i].plot(group["hour"], group["Load"], color=palette[i], alpha=0.1, linewidth=0.3)

        ax[i].set_title(f"Cluster {cluster_label}")
        ax[i].set_xlabel("Ora del giorno")
        ax[i].set_ylabel("Potenza [W]")
        ax[i].set_xticks(range(0, 96, 16))
        ax[i].tick_params(axis='x', labelsize=8)

    # Adding the centroids
    for j, cluster_label in enumerate(cluster_labels):
        cluster_data = centroids[centroids["cluster"] == cluster_label]

        if data_cluster[data_cluster["cluster"] == cluster_label].shape[0] > 96:
            cluster_lower_bound = q1[q1["cluster"] == cluster_label]
            cluster_upper_bound = q3[q3["cluster"] == cluster_label]

        color = palette[j]
        ax[j].plot(cluster_data.set_index("hour")["Load"], color=color, label="Centroid")
        if data_cluster[data_cluster["cluster"] == cluster_label].shape[0] > 96:
            ax[j].fill_between(cluster_lower_bound["hour"], cluster_lower_bound.set_index("hour")["Load"],
                               cluster_upper_bound.set_index("hour")["Load"], color=color, alpha=0.3)

    # Adding the anomalous profiles
    if "Anomalous" in data_cluster["cluster"].unique():
        data_anomalous = data_cluster[data_cluster["cluster"] == "Anomalous"]
        data_anomalous_grouped = data_anomalous.groupby("date")
        for date, group in data_anomalous_grouped:
            ax[j + 1].plot(group["hour"], group["Load"], color='red', alpha=0.3, linewidth=0.5)
    ax[j + 1].set_title("Anomalous")
    ax[j + 1].set_xlabel("Ora del giorno")
    ax[j + 1].set_ylabel("Potenza [W]")
    ax[j + 1].set_xticks(range(0, 96, 16))
    ax[j + 1].tick_params(axis='x', labelsize=8)

    plt.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=4)
    plt.suptitle(f"Profili di carico divisi per cluster per l'utente {building.building_info['name']}",
                 fontsize=16, fontweight='bold')
    plt.savefig(
        os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"load_profiles_{building.building_info['id']}.png"))


def plot_load_profiles_aggregate(aggregate: Aggregate, cluster=None) -> None:
    """
    Grafica i profili di carico di un aggregato dividendoli per cluster. Salva la figura in 'figures/benchmarking'.

    Args:
        aggregate (Aggregate): Oggetto Aggregate.
        cluster (pd.DataFrame): DataFrame con i cluster assegnati agli utenti. Se None, carica il file cluster_{aggregate.name}_assigned.csv.

    Returns:
        None
    """

    if not os.path.exists(os.path.join(PROJECT_ROOT, "results", "benchmarking",
                                       f"cluster_{aggregate.name}_assigned.csv")) and cluster is None:
        raise FileNotFoundError(
            f"File cluster_{aggregate.name}_assigned.csv not found in results. Run assign_cluster first or purchase the cluster file.")
    elif cluster is None:
        cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"cluster_{aggregate.name}_assigned.csv"))

    if not os.path.exists(
            os.path.join(PROJECT_ROOT, "results", "benchmarking", f"medioid_{aggregate.name}_consumer.csv")):
        raise FileNotFoundError(
            f"File medioid_{aggregate.name}_consumer.csv not found in results. Run calculate_medioids_profile first.")
    if not os.path.exists(
            os.path.join(PROJECT_ROOT, "results", "benchmarking", f"medioid_{aggregate.name}_prosumer.csv")):
        raise FileNotFoundError(
            f"File medioid_{aggregate.name}_prosumer.csv not found in results. Run calculate_medioids_profile first.")

    cluster["date"] = pd.to_datetime(cluster["date"]).dt.date
    cluster = cluster[cluster["cluster"] != "Anomalous"]

    cluster_consumer = cluster[cluster["user_type"] == "consumer"]
    cluster_labels_consumer = cluster_consumer["cluster"].unique()
    cluster_labels_consumer.sort()

    cluster_prosumer = cluster[cluster["user_type"] != "consumer"]
    cluster_labels_prosumer = cluster_prosumer["cluster"].unique()
    cluster_labels_prosumer.sort()

    building_data_list = []
    for building in aggregate.buildings:
        data = building.energy_data.data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["hour"] = data["timestamp"].dt.strftime("%H:%M")
        data["date"] = data["timestamp"].dt.date
        data["building_name"] = building.building_info["name"]
        data["Load_norm"] = data["Load"] / data.groupby("date")["Load"].transform("max")

        cluster_user = cluster[cluster["building_name"] == building.building_info["name"]]
        data = pd.merge(data, cluster_user, on="date", how="inner")
        building_data_list.append(data)

    # Calculate the centroids
    medioids = pd.concat([
        pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"medioid_{aggregate.name}_consumer.csv"), index_col=0),
        pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"medioid_{aggregate.name}_prosumer.csv"), index_col=0),
    ]).reset_index(names="cluster")
    medioids = medioids.melt(id_vars="cluster", var_name="hour", value_name="Load_norm")

    # Calculate upper_bound and lower_bound as 1 std on each hour
    total_load_profiles = pd.concat(building_data_list)
    total_load_profiles_std = total_load_profiles[["hour", "Load_norm", "cluster"]].groupby(
        ["hour", "cluster"]).std().reset_index()

    upper_bound = medioids.copy()
    upper_bound["Load_norm"] = (upper_bound["Load_norm"] + total_load_profiles_std["Load_norm"]).clip(upper=1)

    lower_bound = medioids.copy()
    lower_bound["Load_norm"] = (lower_bound["Load_norm"] - total_load_profiles_std["Load_norm"])
    # Clip the lower bound to the minimum value in that cluster
    lower_bound = lower_bound.merge(total_load_profiles.groupby("cluster")["Load_norm"].min().reset_index(),
                                    on="cluster")
    lower_bound["Load_norm"] = lower_bound[["Load_norm_x", "Load_norm_y"]].max(axis=1)
    lower_bound.drop(columns=["Load_norm_x", "Load_norm_y"], inplace=True)

    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=cluster_labels_consumer.size, sharey=True)
    palette = sns.color_palette("colorblind", cluster_labels_consumer.size)

    for building in building_data_list:
        for i, cluster_label in enumerate(cluster_labels_consumer):
            data_cluster = building[building["cluster"] == cluster_label]
            data_cluster_grouped = data_cluster.groupby("date")

            for date, group in data_cluster_grouped:
                ax[i].plot(group["hour"], group["Load_norm"], color=palette[i], alpha=0.1, linewidth=0.1)
            ax[i].set_title(f"Cluster {cluster_label}")
            ax[i].set_xlabel("Ora del giorno")
            ax[i].set_ylabel("Potenza normalizzata [-]")
            ax[i].set_xticks(range(0, 96, 16))
            ax[i].tick_params(axis='x', labelsize=9)

    for j, cluster_label in enumerate(cluster_labels_consumer):
        cluster_data = medioids[medioids["cluster"] == cluster_label]
        cluster_lower_bound = lower_bound[lower_bound["cluster"] == cluster_label]
        cluster_upper_bound = upper_bound[upper_bound["cluster"] == cluster_label]

        color = palette[j]
        ax[j].plot(cluster_data.set_index("hour")["Load_norm"], color=color, label="Centroid Consumer")
        ax[j].fill_between(cluster_lower_bound["hour"], cluster_lower_bound.set_index("hour")["Load_norm"],
                           cluster_upper_bound.set_index("hour")["Load_norm"], color=color, alpha=0.3)

    plt.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=4)
    plt.suptitle(f"Profili di carico dei CONSUMER nei cluster per {aggregate.name.title()}", fontsize=16,
                 fontweight='bold')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"load_profiles_{aggregate.name}_consumer.png"))

    # Same figure but showing the centroid
    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=cluster_labels_consumer.size, sharey=True)
    palette = sns.color_palette("colorblind", cluster_labels_consumer.size)

    for building in building_data_list:
        for i, cluster_label in enumerate(cluster_labels_consumer):
            data_cluster = building[building["cluster"] == cluster_label]
            data_cluster_grouped = data_cluster.groupby("date")

            for date, group in data_cluster_grouped:
                ax[i].plot(group["hour"], group["Load_norm"], color=palette[i], alpha=0.1, linewidth=0.1)
            ax[i].set_title(f"Cluster {cluster_label}")
            ax[i].set_xlabel("Ora del giorno")
            ax[i].set_ylabel("Potenza normalizzata [-]")
            ax[i].set_xticks(range(0, 96, 16))
            ax[i].tick_params(axis='x', labelsize=9)

    centroids = total_load_profiles.groupby(["hour", "cluster"])["Load_norm"].mean().reset_index()
    q1 = total_load_profiles.groupby(["hour", "cluster"])["Load_norm"].quantile(0.25).reset_index()
    q3 = total_load_profiles.groupby(["hour", "cluster"])["Load_norm"].quantile(0.75).reset_index()

    for j, cluster_label in enumerate(cluster_labels_consumer):
        cluster_data = centroids[centroids["cluster"] == cluster_label]
        cluster_lower_bound = q1[q1["cluster"] == cluster_label]
        cluster_upper_bound = q3[q3["cluster"] == cluster_label]

        color = palette[j]
        ax[j].plot(cluster_data.set_index("hour")["Load_norm"], color=color, label="Centroid Consumer")
        ax[j].fill_between(cluster_lower_bound["hour"], cluster_lower_bound.set_index("hour")["Load_norm"],
                           cluster_upper_bound.set_index("hour")["Load_norm"], color=color, alpha=0.3)

    plt.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=4)
    plt.suptitle(f"Profili di carico dei CONSUMER nei cluster per {aggregate.name.title()}", fontsize=16,
                 fontweight='bold')
    plt.savefig(
        os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"load_profiles_{aggregate.name}_consumer_centroids.png"))

    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=cluster_labels_prosumer.size, sharey=True)
    palette = sns.color_palette("colorblind", cluster_labels_prosumer.size)

    for building in building_data_list:
        for i, cluster_label in enumerate(cluster_labels_prosumer):
            data_cluster = building[building["cluster"] == cluster_label]
            data_cluster_grouped = data_cluster.groupby("date")

            for date, group in data_cluster_grouped:
                ax[i].plot(group["hour"], group["Load_norm"], color=palette[i], alpha=0.1, linewidth=0.1)
            ax[i].set_title(f"Cluster {cluster_label}")
            ax[i].set_xlabel("Ora del giorno")
            ax[i].set_ylabel("Potenza normalizzata [-]")
            ax[i].set_xticks(range(0, 96, 16))
            ax[i].tick_params(axis='x', labelsize=9)

    for j, cluster_label in enumerate(cluster_labels_prosumer):
        cluster_data = medioids[medioids["cluster"] == cluster_label]
        cluster_lower_bound = lower_bound[lower_bound["cluster"] == cluster_label]
        cluster_upper_bound = upper_bound[upper_bound["cluster"] == cluster_label]

        color = palette[j]
        ax[j].plot(cluster_data.set_index("hour")["Load_norm"], color=color, label="Centroid Consumer")
        ax[j].fill_between(cluster_lower_bound["hour"], cluster_lower_bound.set_index("hour")["Load_norm"],
                           cluster_upper_bound.set_index("hour")["Load_norm"], color=color, alpha=0.3)

    plt.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=4)
    plt.suptitle(f"Profili di carico dei PROSUMER nei cluster per {aggregate.name.title()}", fontsize=16,
                 fontweight='bold')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"load_profiles_{aggregate.name}_prosumer.png"))

    # Same figure but showing the centroid
    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=cluster_labels_prosumer.size, sharey=True)
    palette = sns.color_palette("colorblind", cluster_labels_prosumer.size)

    for building in building_data_list:
        for i, cluster_label in enumerate(cluster_labels_prosumer):
            data_cluster = building[building["cluster"] == cluster_label]
            data_cluster_grouped = data_cluster.groupby("date")

            for date, group in data_cluster_grouped:
                ax[i].plot(group["hour"], group["Load_norm"], color=palette[i], alpha=0.1, linewidth=0.1)
            ax[i].set_title(f"Cluster {cluster_label}")
            ax[i].set_xlabel("Ora del giorno")
            ax[i].set_ylabel("Potenza normalizzata [-]")
            ax[i].set_xticks(range(0, 96, 16))
            ax[i].tick_params(axis='x', labelsize=9)

    centroids = total_load_profiles.groupby(["hour", "cluster"])["Load_norm"].mean().reset_index()
    q1 = total_load_profiles.groupby(["hour", "cluster"])["Load_norm"].quantile(0.25).reset_index()
    q3 = total_load_profiles.groupby(["hour", "cluster"])["Load_norm"].quantile(0.75).reset_index()

    for j, cluster_label in enumerate(cluster_labels_prosumer):
        cluster_data = centroids[centroids["cluster"] == cluster_label]
        cluster_lower_bound = q1[q1["cluster"] == cluster_label]
        cluster_upper_bound = q3[q3["cluster"] == cluster_label]

        color = palette[j]
        ax[j].plot(cluster_data.set_index("hour")["Load_norm"], color=color, label="Centroid Consumer")
        ax[j].fill_between(cluster_lower_bound["hour"], cluster_lower_bound.set_index("hour")["Load_norm"],
                           cluster_upper_bound.set_index("hour")["Load_norm"], color=color, alpha=0.3)

    plt.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=4)
    plt.suptitle(f"Profili di carico dei PROSUMER nei cluster per {aggregate.name.title()}", fontsize=16,
                 fontweight='bold')
    plt.savefig(
        os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"load_profiles_{aggregate.name}_prosumer_centroids.png"))


def plot_cluster_percentage(aggregate: Aggregate, cluster=None) -> None:
    """
    Grafica la percentuale di profili in ogni cluster per ogni utente dell'aggregato. Salva la figura in 'figures/benchmarking'.

    Args:
        aggregate (Aggregate): Oggetto Aggregate.
        cluster (pd.DataFrame): DataFrame con i cluster assegnati agli utenti. Se None, carica il file cluster_{aggregate.name}_assigned.csv.

    Returns:
        None
    """

    if cluster is None:
        if not os.path.exists(
                os.path.join(PROJECT_ROOT, "results", "benchmarking", f"cluster_{aggregate.name}_assigned.csv")):
            raise FileNotFoundError(
                f"File cluster_{aggregate.name}_assigned.csv not found in results. Run assign_cluster first or purchase the cluster file.")
        else:
            cluster = pd.read_csv(
                os.path.join(PROJECT_ROOT, "results", "benchmarking", f"cluster_{aggregate.name}_assigned.csv"))

    consumer_cluster = cluster[cluster["user_type"] == "consumer"]
    consumer_cluster_grouped = consumer_cluster.groupby(["building_name", "cluster"]).size().reset_index(name="count")
    consumer_cluster_grouped["percentage"] = consumer_cluster_grouped.groupby("building_name")["count"].transform(
        lambda x: x / x.sum() * 100)

    df_cluster_percentage = consumer_cluster_grouped.pivot(index="building_name", columns="cluster",
                                                           values="percentage")
    df_cluster_percentage.fillna(0, inplace=True)
    df_cluster_percentage = df_cluster_percentage.round(1)

    if 'Anomalous' in df_cluster_percentage.columns:
        palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns) - 1)
        palette.insert(0, (1, 0, 0))
    else:
        palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns))

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = df_cluster_percentage.plot(kind="barh", stacked=True, ax=ax, color=palette)
    ax.set_xlabel("Percentuale [%]", fontsize=14)
    ax.set_ylabel("Edificio", fontsize=14)
    ax.set_title(f"Percentuale di profili di carico dei CONSUMER nei cluster per {aggregate.name.title()}",
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)
    for container in bars.containers:
        for rect in container:
            width = rect.get_width()
            if width > 0:
                x = rect.get_x() + width / 2
                y = rect.get_y() + rect.get_height() / 2
                percentage = f"{width:.1f}%"
                ax.text(x, y, percentage, ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(df_cluster_percentage.columns),
              fancybox=True, shadow=True)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(
        os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"cluster_percentage_{aggregate.name}_consumer.png"))

    producer_cluster = cluster[cluster["user_type"] != "consumer"]
    producer_cluster_grouped = producer_cluster.groupby(["building_name", "cluster"]).size().reset_index(name="count")
    producer_cluster_grouped["percentage"] = producer_cluster_grouped.groupby("building_name")["count"].transform(
        lambda x: x / x.sum() * 100)

    df_cluster_percentage = producer_cluster_grouped.pivot(index="building_name", columns="cluster",
                                                           values="percentage")
    df_cluster_percentage.fillna(0, inplace=True)
    df_cluster_percentage = df_cluster_percentage.round(1)

    if 'Anomalous' in df_cluster_percentage.columns:
        palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns) - 1)
        # Append the red color to the palette at first position
        palette.insert(0, (1, 0, 0))
    else:
        palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns))
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = df_cluster_percentage.plot(kind="barh", stacked=True, ax=ax, color=palette)
    ax.set_xlabel("Percentuale [%]", fontsize=14)
    ax.set_ylabel("Edificio", fontsize=14)
    ax.set_title(f"Percentuale di profili di carico dei PROSUMER nei cluster per {aggregate.name.title()}",
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)
    for container in bars.containers:
        for rect in container:
            width = rect.get_width()
            if width > 0:
                x = rect.get_x() + width / 2
                y = rect.get_y() + rect.get_height() / 2
                percentage = f"{width:.1f}%"
                ax.text(x, y, percentage, ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(df_cluster_percentage.columns),
              fancybox=True, shadow=True)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(
        os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"cluster_percentage_{aggregate.name}_prosumer.png"))


def plot_cluster_population(aggregate: Aggregate, cluster=None) -> None:
    """
    Grafica la popolazione dei cluster per consumer e prosumer con un bar plot diviso per cluster e ordinato dal maggiore
    al minore. Salva la figura in 'figures/benchmarking'.

    Args:
        aggregate (Aggregate): Oggetto Aggregate.
        cluster (pd.DataFrame): DataFrame con i cluster assegnati agli utenti. Se None, carica il file cluster_{aggregate.name}_assigned.csv.

    Returns:
        None
    """

    if cluster is None:
        if not os.path.exists(
                os.path.join(PROJECT_ROOT, "results", "benchmarking", f"cluster_{aggregate.name}_assigned.csv")):
            raise FileNotFoundError(
                f"File cluster_{aggregate.name}_assigned.csv not found in results. Run assign_cluster first or purchase the cluster file.")
        else:
            cluster = pd.read_csv(
                os.path.join(PROJECT_ROOT, "results", "benchmarking", f"cluster_{aggregate.name}_assigned.csv"))


    cluster = cluster[cluster["cluster"] != "Anomalous"]

    cluster_consumer = cluster[cluster["user_type"] == "consumer"]
    cluster_count = cluster_consumer.groupby("cluster").size().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=cluster_count.values, y=cluster_count.index, ax=ax, palette="colorblind")
    ax.set_xlabel("Numero di profili", fontsize=14)
    ax.set_ylabel("Cluster", fontsize=14)
    plt.title(f"Popolazione dei cluster per i CONSUMER in {aggregate.name.title()}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"cluster_population_{aggregate.name}_consumer.png"))

    cluster_prosumer = cluster[cluster["user_type"] != "consumer"]
    cluster_count = cluster_prosumer.groupby("cluster").size().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=cluster_count.values, y=cluster_count.index, ax=ax, palette="colorblind")
    ax.set_xlabel("Numero di profili", fontsize=14)
    ax.set_ylabel("Cluster", fontsize=14)
    plt.title(f"Popolazione dei cluster per i PROSUMER in {aggregate.name.title()}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "benchmarking", f"cluster_population_{aggregate.name}_prosumer.png"))


def plot_feature_distribution(aggregate: Aggregate, cluster=None) -> None:
    """
    Plotta la distribuzione degli indicatori di forma per i consumer e i prosumer. Salva la figura in 'figures/benchmarking'.

    Args:
        aggregate (Aggregate): Oggetto Aggregate.
        cluster (pd.DataFrame): DataFrame con i cluster assegnati agli utenti. Se None, carica il file cluster_{aggregate.name}_assigned.csv.

    Returns:
        None
    """

    if cluster is None:
        if not os.path.exists(
                os.path.join(PROJECT_ROOT, "results", "benchmarking", f"cluster_{aggregate.name}_assigned.csv")):
            raise FileNotFoundError(
                f"File cluster_{aggregate.name}_assigned.csv not found in results. Run assign_cluster first or purchase the cluster file.")
        else:
            cluster = pd.read_csv(
                os.path.join(PROJECT_ROOT, "results", "benchmarking", f"cluster_{aggregate.name}_assigned.csv"))

    cluster = cluster[cluster["cluster"] != "Anomalous"]
    cluster["date"] = pd.to_datetime(cluster["date"]).dt.date

    building_sf_dict = {}
    for building in aggregate.buildings:
        data = building.energy_data.data.copy()
        data.set_index("timestamp", inplace=True)
        data = data["Load"].clip(lower=0)
        data = data.resample("1H").mean()
        df_sf = calculate_shape_factor(data)
        df_sf["building_name"] = building.building_info["name"]
        df_sf["user_type"] = building.building_info["user_type"]
        building_sf_dict[building.building_info["name"]] = df_sf

    df_sf_total = pd.concat(building_sf_dict.values())
    df_sf_total.reset_index(inplace=True, names="date")

    data = pd.merge(df_sf_total,
                    cluster[["date", "cluster", "building_name"]],
                    on=["date", "building_name"],
                    how="inner")

    data_consumer = data[data["user_type"] == "consumer"]
    data_consumer.sort_values("cluster", inplace=True)
    consumer_labels = data_consumer["cluster"].unique()
    consumer_labels.sort()

    fig, axes = plt.subplots(figsize=(12, 8), nrows=4, ncols=4)
    for i, column in enumerate(data_consumer.columns[1:-3]):
        ax = axes[i // 4, i % 4]
        sns.kdeplot(data=data_consumer, x=column, hue="cluster", ax=ax, fill=True, palette="colorblind",
                    common_norm=False, legend=False)
        # ax.set_title(f"Distribuzione di {column}", fontsize=14)
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(data_consumer["cluster"].unique()),
               fancybox=True, shadow=True, labels=consumer_labels)
    plt.suptitle("Distribuzione degli indicatori di forma per i CONSUMER", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0.05, 1, 0.92))
    plt.savefig(
        os.path.join(PROJECT_ROOT, "figures", "benchmarking", "feature_distribution", f"dist_{aggregate.name}_consumer.png"))

    fig, axes = plt.subplots(figsize=(12, 10), nrows=4, ncols=4)
    for i, column in enumerate(data_consumer.columns[1:-3]):
        ax = axes[i // 4, i % 4]
        sns.violinplot(data=data_consumer, y="cluster", x=column, ax=ax, palette="colorblind", inner="box",
                       common_norm=True, legend=False, alpha=0.7)
        ax.set_xlabel(column)
        ax.set_ylabel("Cluster")

    palette = sns.color_palette("colorblind", n_colors=len(consumer_labels))
    palette = [(r, g, b, 0.7) for r, g, b in palette]
    handles = [Line2D([0], [0], color=palette[i], lw=4) for i in range(len(consumer_labels))]
    fig.legend(handles=handles, labels=consumer_labels.tolist(), loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(consumer_labels), fancybox=True, shadow=True)
    plt.suptitle("Distribuzione degli indicatori di forma per i CONSUMER", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0.05, 1, 0.92))
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "benchmarking", "feature_distribution",
                             f"violin_{aggregate.name}_consumer.png"))

    data_prosumer = data[data["user_type"] != "consumer"]
    data_prosumer.sort_values("cluster", inplace=True)
    prosumer_labels = data_prosumer["cluster"].unique()
    prosumer_labels.sort()

    fig, axes = plt.subplots(figsize=(12, 8), nrows=4, ncols=4)
    for i, column in enumerate(data_prosumer.columns[1:-3]):
        ax = axes[i // 4, i % 4]
        sns.kdeplot(data=data_prosumer, x=column, hue="cluster", ax=ax, fill=True, palette="colorblind",
                    common_norm=False, legend=False)
        # ax.set_title(f"Distribuzione di {column}", fontsize=14)
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(data_prosumer["cluster"].unique()),
               fancybox=True, shadow=True, labels=prosumer_labels)
    plt.suptitle("Distribuzione degli indicatori di forma per i PROSUMER", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0.05, 1, 0.92))
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "benchmarking", "feature_distribution", f"dist_{aggregate.name}_prosumer.png"))

    fig, axes = plt.subplots(figsize=(12, 10), nrows=4, ncols=4)
    for i, column in enumerate(data_prosumer.columns[1:-3]):
        ax = axes[i // 4, i % 4]
        sns.violinplot(data=data_prosumer, y="cluster", x=column, ax=ax, palette="colorblind", inner="box",
                       common_norm=True, legend=False, alpha=0.7)
        ax.set_xlabel(column)
        ax.set_ylabel("Cluster")

    palette = sns.color_palette("colorblind", n_colors=len(prosumer_labels))
    palette = [(r, g, b, 0.7) for r, g, b in palette]
    handles = [Line2D([0], [0], color=palette[i], lw=4) for i in range(len(consumer_labels))]
    fig.legend(handles=handles, labels=prosumer_labels.tolist(), loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(prosumer_labels), fancybox=True, shadow=True)
    plt.suptitle("Distribuzione degli indicatori di forma per i PROSUMER", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0.05, 1, 0.92))
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "benchmarking", "feature_distribution",
                             f"violin_{aggregate.name}_prosumer.png"))