import matplotlib.pyplot as plt
import pandas as pd
import os
from settings import PROJECT_ROOT
from src.building import Building, load_anguillara, load_garda
from src.benchmarking.utils import find_medioid_and_quartiles
import seaborn as sns


def plot_silhouette_scores(scores: pd.DataFrame):
    """
    Grafica il silhouette score per un numero di cluster variabile
    :param scores: dataframe con silhouette score per un numero di cluster variabile
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


def plot_load_profiles_user(user_id: str, aggregate: str):
    """
    Grafica i profili di carico di un utente dividendoli per cluster
    :param user_id: id dell'utente
    :param aggregate: nome dell'aggregato ("anguillara" o "garda")
    """

    building = Building(user_id, aggregate)
    data = building.energy_meter.data

    cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"cluster_{aggregate}_assigned.csv"))
    cluster_user = cluster[cluster["user_id"] == user_id]
    cluster_user["date"] = pd.to_datetime(cluster_user["date"]).dt.date

    # Merge data with cluster user on the date
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date

    data_cluster = pd.merge(data[["timestamp", "date", "Load"]], cluster_user[["date", "cluster"]], on="date", how="inner")
    data_cluster["hour"] = data_cluster["timestamp"].dt.strftime("%H:%M")

    cluster_labels = cluster.loc[cluster["user_type"] == building.building_info["user_type"], "cluster"].unique()
    cluster_labels.sort()

    data_cluster_pivot = data_cluster.pivot(index=["date", "cluster"], columns="hour", values="Load").reset_index().drop(columns=["date"])

    centroids = data_cluster.groupby(["hour", "cluster"])["Load"].mean().reset_index()
    centroids_pivot = centroids.pivot(index="cluster", columns="hour", values="Load")

    medioid, q1, q3 = find_medioid_and_quartiles(data_cluster_pivot, centroids_pivot)
    medioid = medioid.reset_index(names=["cluster"]).melt(id_vars="cluster", var_name="hour", value_name="Load")

    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=cluster_labels.size, sharey=True)

    palette = sns.color_palette("colorblind", cluster_labels.size)

    for i, cluster_label in enumerate(cluster_labels):
        data_cluster_viz = data_cluster[data_cluster["cluster"] == cluster_label]
        data_cluster_viz_grouped = data_cluster_viz.groupby("date")

        for date, group in data_cluster_viz_grouped:
            ax[i].plot(group["hour"], group["Load"], color=palette[i], alpha=0.1, linewidth=0.3)

        ax[i].set_title(f"Cluster {cluster_label}")
        ax[i].set_xlabel("Ora del giorno")
        ax[i].set_ylabel("Potenza [kW]")
        ax[i].set_xticks(range(0, 96, 16))
        ax[i].tick_params(axis='x', labelsize=9)

    # Calculate upper_bound and lower_bound as 1 std on each hour
    total_load_profiles = data_cluster[["hour", "Load", "cluster"]]
    total_load_profiles_std = total_load_profiles.groupby(["hour", "cluster"]).std().reset_index()

    upper_bound = medioid.copy()
    upper_bound["Load"] = (upper_bound["Load"] + total_load_profiles_std["Load"]).clip()

    lower_bound = medioid.copy()
    lower_bound["Load"] = (lower_bound["Load"] - total_load_profiles_std["Load"])
    # Clip
    lower_bound = lower_bound.merge(total_load_profiles.groupby("cluster")["Load"].min().reset_index(), on="cluster")
    lower_bound["Load"] = lower_bound[["Load_x", "Load_y"]].max(axis=1)
    lower_bound.drop(columns=["Load_x", "Load_y"], inplace=True)

    for j, cluster_label in enumerate(cluster_labels):
        cluster_data = medioid[medioid["cluster"] == cluster_label]
        cluster_lower_bound = lower_bound[lower_bound["cluster"] == cluster_label]
        cluster_upper_bound = upper_bound[upper_bound["cluster"] == cluster_label]

        color = palette[j]
        ax[j].plot(cluster_data.set_index("hour")["Load"], color=color, label="Centroid")
        ax[j].fill_between(cluster_lower_bound["hour"], cluster_lower_bound.set_index("hour")["Load"],
                           cluster_upper_bound.set_index("hour")["Load"], color=color, alpha=0.3)

    plt.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=4)
    plt.suptitle(f"Profili di carico divisi per cluster per l'utente {building.building_info['name']}",
                    fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "clustering", f"load_profiles_{user_id}.png"))


def plot_load_profiles_aggregate(aggregate: str):
    """
    Grafica i profili di carico di un aggregato dividendoli per cluster
    :param aggregate: nome dell'aggregato ("anguillara" o "garda")
    """

    cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"cluster_{aggregate}_assigned.csv"))
    cluster["date"] = pd.to_datetime(cluster["date"]).dt.date

    cluster_consumer = cluster[cluster["user_type"] == "consumer"]
    cluster_labels_consumer = cluster_consumer["cluster"].unique()
    cluster_labels_consumer.sort()

    cluster_prosumer = cluster[cluster["user_type"] != "consumer"]
    cluster_labels_prosumer = cluster_prosumer["cluster"].unique()
    cluster_labels_prosumer.sort()

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
        data = pd.merge(data, cluster_user, on="date", how="inner")
        building_data_list.append(data)

    # Calculate the centroids
    medioids = pd.concat([
        pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"medioid_{aggregate}_consumer.csv"), index_col=0),
        pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"medioid_{aggregate}_prosumer.csv"), index_col=0),
    ]).reset_index(names="cluster")
    medioids = medioids.melt(id_vars="cluster", var_name="hour", value_name="Load_norm")

    # Calculate upper_bound and lower_bound as 1 std on each hour
    total_load_profiles = pd.concat(building_data_list)
    total_load_profiles_std = total_load_profiles[["hour", "Load_norm", "cluster"]].groupby(["hour", "cluster"]).std().reset_index()

    upper_bound = medioids.copy()
    upper_bound["Load_norm"] = (upper_bound["Load_norm"] + total_load_profiles_std["Load_norm"]).clip(upper=1)

    lower_bound = medioids.copy()
    lower_bound["Load_norm"] = (lower_bound["Load_norm"] - total_load_profiles_std["Load_norm"])
    # Clip the lower bound to the minimum value in that cluster
    lower_bound = lower_bound.merge(total_load_profiles.groupby("cluster")["Load_norm"].min().reset_index(), on="cluster")
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
    plt.suptitle(f"Profili di carico dei CONSUMER nei cluster per {aggregate.title()}", fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "clustering", f"load_profiles_{aggregate}_consumer.png"))

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
    plt.suptitle(f"Profili di carico dei PROSUMER nei cluster per {aggregate.title()}", fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "clustering", f"load_profiles_{aggregate}_prosumer.png"))


def plot_cluster_percentage(aggregate: str):
    """
    Grafica la percentuale di profili in ogni cluster per ogni utente dell'aggregato
    :param aggregate: nome dell'aggregato ("anguillara" o "garda")
    """

    cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", f"cluster_{aggregate}_assigned.csv"))

    consumer_cluster = cluster[cluster["user_type"] == "consumer"]
    consumer_cluster_grouped = consumer_cluster.groupby(["building_name", "cluster"]).size().reset_index(name="count")
    consumer_cluster_grouped["percentage"] = consumer_cluster_grouped.groupby("building_name")["count"].transform(
        lambda x: x / x.sum() * 100)

    df_cluster_percentage = consumer_cluster_grouped.pivot(index="building_name", columns="cluster", values="percentage")
    df_cluster_percentage.fillna(0, inplace=True)
    df_cluster_percentage = df_cluster_percentage.round(1)

    palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns))
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = df_cluster_percentage.plot(kind="barh", stacked=True, ax=ax, color=palette)
    ax.set_xlabel("Percentuale [%]", fontsize=14)
    ax.set_ylabel("Edificio", fontsize=14)
    ax.set_title(f"Percentuale di profili di carico dei CONSUMER nei cluster per {aggregate.title()}",
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
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "clustering", f"cluster_percentage_{aggregate}_consumer.png"))

    producer_cluster = cluster[cluster["user_type"] != "consumer"]
    producer_cluster_grouped = producer_cluster.groupby(["building_name", "cluster"]).size().reset_index(name="count")
    producer_cluster_grouped["percentage"] = producer_cluster_grouped.groupby("building_name")["count"].transform(
        lambda x: x / x.sum() * 100)

    df_cluster_percentage = producer_cluster_grouped.pivot(index="building_name", columns="cluster", values="percentage")
    df_cluster_percentage.fillna(0, inplace=True)
    df_cluster_percentage = df_cluster_percentage.round(1)

    palette = sns.color_palette("colorblind", n_colors=len(df_cluster_percentage.columns))
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = df_cluster_percentage.plot(kind="barh", stacked=True, ax=ax, color=palette)
    ax.set_xlabel("Percentuale [%]", fontsize=14)
    ax.set_ylabel("Edificio", fontsize=14)
    ax.set_title(f"Percentuale di profili di carico dei PROSUMER nei cluster per {aggregate.title()}",
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
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "clustering", f"cluster_percentage_{aggregate}_prosumer.png"))


def plot_centroids(aggregate: str):
    pass


plot_load_profiles_user("7436df46-294b-4c97-bd1b-8aaa3aed97c5", "anguillara")